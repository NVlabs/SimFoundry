# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import torch
import torch.nn
from simfoundry.models.text_encoder import TextEncoder
from simfoundry.utils.processing_utils import create_polygon_from_vertices, mask_intersection_area, mask_area
import numpy as np
import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model_builder import build_sam3_video_predictor
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
from skimage import morphology
import os
import cv2
import glob
import hydra
from hydra import initialize_config_dir

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SAM3_ROOT = os.path.join(os.path.dirname(sam3.__file__), "..")


class SAM3(torch.nn.Module):

    def __init__(
        self,
        confidence_threshold=0.50,
        device="cuda",
        video=False,
        n_video_devices=-1,         # Number of devices to use for video prediction, -1 for all avilable
        enable_inst_interactivity=False,
    ):

        super().__init__()

        self.device = device
        self.confidence_threshold = confidence_threshold

        # Load SAM model
        bpe_path = f"{SAM3_ROOT}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        if video:
            assert self.device == "cuda", "Must set device=cuda when using SAM3 video model!"
            gpus_to_use = range(torch.cuda.device_count()) if n_video_devices == -1 else n_video_devices
            model = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
        else:
            raw_model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=enable_inst_interactivity)
            model = Sam3Processor(raw_model, confidence_threshold=confidence_threshold)
        self.model = model
        self.video = video

    def predict_segmentation(self, pil_img, text_prompt):
        assert not self.video, "Image segmentation only available with video=False!"
        # SAM3's fused ViT MLP kernels emit bfloat16 activations while the linear
        # weights remain float32, so inference must run inside a bfloat16 autocast
        # context (matching the official SAM3 examples) to avoid a dtype mismatch.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.model.set_image(pil_img)
            self.model.reset_all_prompts(inference_state)
            result = self.model.set_text_prompt(state=inference_state, prompt=text_prompt)

        # Return / extract values
        # Masks: Shape (N, 1, H, W)
        # Boxes: Shape (N, 4) - xyxy
        # Scores: Shape (N,)
        return result["masks"].cpu().numpy(), result["boxes"].cpu().numpy(), result["scores"].float().cpu().numpy()

    def predict_segmentation_with_point(self, pil_img, point_coords, point_labels=None, multimask_output=False):
        """
        Predict segmentation using point prompts via the SAM3 interactive predictor.

        Args:
            pil_img: PIL Image to segment
            point_coords: List of (x, y) coordinates for point prompts. Can be:
                - Single point: (x, y) or [(x, y)]
                - Multiple points: [(x1, y1), (x2, y2), ...]
            point_labels: Optional list of labels for each point (1 = foreground, 0 = background).
                If None, all points are assumed to be foreground (label=1).
            multimask_output: If True, return multiple mask predictions per point.

        Returns:
            masks: Shape (N, 1, H, W) boolean masks
            boxes: Shape (N, 4) bounding boxes in xyxy format
            scores: Shape (N,) confidence scores
        """
        assert not self.video, "Image segmentation only available with video=False!"

        if isinstance(point_coords, tuple) and len(point_coords) == 2:
            point_coords = [point_coords]
        point_coords = [tuple(p) for p in point_coords]

        if point_labels is None:
            point_labels = [1] * len(point_coords)

        points_np = np.array(point_coords, dtype=np.float32)
        labels_np = np.array(point_labels, dtype=np.int32)

        # Use the main model's backbone to compute image features, then
        # wire them into the interactive predictor (same as Sam3Image.predict_inst).
        # SAM3's fused ViT MLP kernels emit bfloat16 activations while the linear
        # weights remain float32, so inference must run inside a bfloat16 autocast
        # context (matching the official SAM3 examples) to avoid a dtype mismatch.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.model.set_image(pil_img)
            raw_model = self.model.model  # Sam3Image
            predictor = raw_model.inst_interactive_predictor

            backbone_out = inference_state["backbone_out"]["sam2_backbone_out"]
            _, vision_feats, _, _ = predictor.model._prepare_backbone_features(backbone_out)
            vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(1, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1])
            ][::-1]

            predictor._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
            predictor._is_image_set = True
            predictor._orig_hw = [(inference_state["original_height"], inference_state["original_width"])]

            masks, scores, low_res_logits = predictor.predict(
                point_coords=points_np,
                point_labels=labels_np,
                multimask_output=multimask_output,
            )

            predictor._features = None
            predictor._is_image_set = False

        # masks shape from predict: (C, H, W) — add object dimension -> (C, 1, H, W)
        if len(masks.shape) == 3:
            masks = masks[:, np.newaxis, :, :]

        boxes = self._masks_to_boxes(masks)

        return masks, boxes, scores
    
    def _masks_to_boxes(self, masks):
        """
        Compute bounding boxes from binary masks.
        
        Args:
            masks: Shape (N, 1, H, W) or (N, H, W) boolean masks
            
        Returns:
            boxes: Shape (N, 4) bounding boxes in xyxy format
        """
        if len(masks.shape) == 4:
            masks = masks.squeeze(1)  # (N, H, W)
        
        boxes = []
        for mask in masks:
            if mask.sum() == 0:
                boxes.append([0, 0, 0, 0])
                continue
            
            # Find non-zero indices
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            boxes.append([x_min, y_min, x_max, y_max])
        
        return np.array(boxes, dtype=np.float32)

    def predict_video_segmentation(self, video_fpath, text_prompt):
        assert self.video, "Video segmentation only available with video=True!"

        # We can handle either .mp4 files or directory of numbered, ordered frame images

        # Initialize inference session
        response = self.model.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_fpath,
            )
        )
        session_id = response["session_id"]

        # Make sure session is reset so the model is not tracking stale outdated state
        _ = self.model.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

        frame_idx = 0  # add a text prompt on frame 0
        response = self.model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx,
                text=text_prompt,
            )
        )
        out = response["outputs"]

        # now we propagate the outputs from frame 0 to the end of the video and collect all outputs
        outputs_per_frame = self._propagate_in_video(session_id=session_id)

        # Dict, mapping frame idxs to the following:
        # out_obj_ids: (N,)
        # out_probs: (N,)
        # out_boxes_xywh (N, 4)
        # out_binary_masks: (N, H, W)
        # frame_stats: None
        return outputs_per_frame

    def _propagate_in_video(self, session_id):
        # we will just propagate from frame 0 to the end of the video
        outputs_per_frame = {}
        for response in self.model.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]

        return outputs_per_frame

    def prune_redundant_masks(
            self,
            masks,
            boxes_xyxy,
            logits,
            phrases,
            polygon_relative_intersection_threshold=0.97,
            polygon_relative_area_threshold=0.9,
            obj_mask_intersect_area_threshold=0.9,
            duplicate_mask_intersect_area_threshold=0.95,
            boundary_proportion=None,
            text_encoder=None,
            text_encoder_similarity_threshold=0.85,
            verbose=False,
    ):
        # Loop over all boxes and compare them to all other boxes that were detected
        H, W = masks[0].shape

        # We prune a given box if:
        # a) the object category is FLOOR_CATEGORY
        # b) a given bounding box has sufficient overlap with multiple smaller bounding boxes with the same caption (avoid a large mask covering multiple instances of the same category)
        # c) it is near the boundary of the image (if boundary_proportion is not None)

        idxs_to_remove = dict()
        susp_small_large_box_idxs = set()
        for i, (box_a, phrase_a) in enumerate(zip(boxes_xyxy, phrases)):
            lower_a_x, lower_a_y, upper_a_x, upper_a_y = box_a
            polygon_a = create_polygon_from_vertices(
                [(lower_a_x, lower_a_y), (upper_a_x, lower_a_y), (upper_a_x, upper_a_y), (lower_a_x, upper_a_y)])
            for j, (box_b, phrase_b) in enumerate(zip(boxes_xyxy, phrases)):
                # Skip self
                if i == j:
                    continue

                # Check bi-directional phrase overlap -- i.e.: A and B both belong to roughly the same category
                elif phrase_a in phrase_b or phrase_b in phrase_a:
                    lower_b_x, lower_b_y, upper_b_x, upper_b_y = box_b
                    polygon_b = create_polygon_from_vertices(
                        [(lower_b_x, lower_b_y), (upper_b_x, lower_b_y), (upper_b_x, upper_b_y),
                         (lower_b_x, upper_b_y)])

                    # Check if both A and B's polygons intersect sufficiently
                    if polygon_a.intersects(polygon_b):
                        intersect_area = polygon_a.intersection(polygon_b).area
                        if intersect_area / polygon_a.area >= polygon_relative_intersection_threshold or intersect_area / polygon_b.area >= polygon_relative_intersection_threshold:
                            # If the two masks are roughly the same size, remove the smaller one
                            # If one mask is much smaller than the other, remove the larger one
                            if polygon_a.area < polygon_b.area:
                                if polygon_a.area < polygon_relative_area_threshold * polygon_b.area:
                                    susp_small_large_box_idxs.add((i, j))
                                else:
                                    if i not in idxs_to_remove:
                                        idxs_to_remove[i] = "box_xyxy overlap"
                                    # idxs_to_remove.add(i)
                            else:
                                if polygon_b.area < polygon_relative_area_threshold * polygon_a.area:
                                    susp_small_large_box_idxs.add((j, i))
                                else:
                                    if j not in idxs_to_remove:
                                        idxs_to_remove[j] = "box_xyxy overlap"
                                    # idxs_to_remove.add(j)

        # If there are multiple small boxes within a given large boxes, we remove the large box.
        # Otherwise, we remove the small box
        for i, (smaller_idx, larger_idx) in enumerate(susp_small_large_box_idxs):
            for j, (smaller_idx_2, larger_idx_2) in enumerate(susp_small_large_box_idxs):
                # Skip if self
                if i == j:
                    continue
                if larger_idx == larger_idx_2:
                    if larger_idx not in idxs_to_remove:
                        idxs_to_remove[larger_idx] = "large bbox overlapping multiple small bbox"
                    # idxs_to_remove.add(larger_idx)
                    break

        # Prune noise from masks
        for i, mask in enumerate(masks):
            masks[i] = morphology.remove_small_objects(mask, min_size=int(np.sqrt(W * H) / 10.0),
                                                               connectivity=1)

        # Remove a given mask if it has sufficient overlap with a larger mask (avoid having masks for a proportion of a whole object)
        mask_areas = [mask_area(mask) for mask in masks]
        for i, (mask_a, mask_area_a, phrase_a, logit_a) in enumerate(zip(masks, mask_areas, phrases, logits)):
            if i in idxs_to_remove:
                continue

            for j, (mask_b, mask_area_b, phrase_b, logit_b) in enumerate(zip(masks, mask_areas, phrases, logits)):
                # Skip self and objects already deleted
                if (i == j) or (j in idxs_to_remove) or (i in idxs_to_remove):
                    continue

                # Check if the two masks are nearly identical -- if so, prune the one with the lower logit
                inter_area = mask_intersection_area(mask_a, mask_b)
                if (inter_area / mask_area_a >= duplicate_mask_intersect_area_threshold) and \
                    (inter_area / mask_area_b >= duplicate_mask_intersect_area_threshold):
                    # Remove the mask with the lower logit
                    idx_to_remove = i if logit_a < logit_b else j
                    idxs_to_remove[idx_to_remove] = "duplicate mask with same pixels"
                    continue

                # Skip if phrases do not overlap by cosine threshold if using CLIP (i.e.: these are clearly different objects)
                # if clip is not None:
                #     assert isinstance(clip, CLIPEncoder)
                #     f0, f1 = clip.get_text_features([phrase_a, phrase_b])
                #     cos_sim = np.dot(f0, f1) / (np.linalg.norm(f0) * np.linalg.norm(f1))
                #     if cos_sim < clip_cosine_threshold:
                #         continue
                #     print(f"Detected similar phrases [{phrase_a} // {phrase_b}], checking for mask overlap...")
                if text_encoder is not None:
                    assert isinstance(text_encoder, TextEncoder)
                    text_features = text_encoder.get_text_features([phrase_a, phrase_b])
                    cos_sim = text_encoder.get_similarity_matrix(text_features, text_features)[0, 1]
                    if cos_sim < text_encoder_similarity_threshold:
                        continue
                    print(f"Detected similar phrases [{phrase_a} // {phrase_b}], checking for mask overlap...")
                # if not (phrase_a in phrase_b or phrase_b in phrase_a):
                #     continue
                min_area = min(mask_area_a, mask_area_b)

                if inter_area > obj_mask_intersect_area_threshold * min_area:
                    if mask_area_a > mask_area_b:
                        print(f"Pruning {phrase_b} because of sufficient overlap")
                        if j not in idxs_to_remove:
                            idxs_to_remove[j] = "mask pixels proportion overlap"
                            continue
                        # idxs_to_remove.add(j)
                    else:
                        print(f"Pruning {phrase_a} because of sufficient overlap")
                        if i not in idxs_to_remove:
                            idxs_to_remove[i] = "mask pixels proportion overlap"
                            continue
                        # idxs_to_remove.add(i)
                        break

        if boundary_proportion is not None:
            # Skip this mask if any pixel in the mask is near the edge of the image
            for i, mask in enumerate(masks):
                h_idxs, w_idxs = np.where(mask)
                if np.any((h_idxs < H * boundary_proportion) |
                          (h_idxs > H * (1 - boundary_proportion)) |
                          (w_idxs < W * boundary_proportion) |
                          (w_idxs > W * (1 - boundary_proportion))
                          ):
                    if i not in idxs_to_remove:
                        idxs_to_remove[i] = "near image boundary"
                    # idxs_to_remove.add(i)

        pruned_masks = deepcopy(masks)
        pruned_boxes_xyxy = deepcopy(boxes_xyxy)
        pruned_logits = deepcopy(logits)
        pruned_phrases = deepcopy(phrases)

        # Delete the redundancies
        for idx in sorted(idxs_to_remove.keys(), reverse=True):
            reason = idxs_to_remove[idx]
            pruned_boxes_xyxy = np.concatenate((pruned_boxes_xyxy[:idx], pruned_boxes_xyxy[idx + 1:]))
            pruned_logits = np.concatenate((pruned_logits[:idx], pruned_logits[idx + 1:]))
            del pruned_masks[idx]
            if verbose:
                print(f"remove idx:{idx} phrase:{phrases[idx]}, reason: {reason}")
            del pruned_phrases[idx]

        if verbose:
            print(f"idxs to remove: {idxs_to_remove}")

        return pruned_masks, pruned_boxes_xyxy, pruned_logits, pruned_phrases

    def prune_masks_below_surface(
            self,
            masks,
            boxes_xyxy,
            logits,
            phrases,
            pc,
            valid_mask=None,
            z_threshold=0.0,
            z_threshold_proportion=0.1,
            verbose=False,
    ):
        pruned_masks = deepcopy(masks)
        pruned_boxes_xyxy = deepcopy(boxes_xyxy)
        pruned_logits = deepcopy(logits)
        pruned_phrases = deepcopy(phrases)

        idxs_to_remove = set()
        for idx, mask in enumerate(masks):
            valid_pixels = mask & valid_mask if valid_mask is not None else mask
            valid_points = pc[valid_pixels].reshape(-1, 3)
            if (np.sum(valid_points[:, 2] < z_threshold) / len(valid_points)) > z_threshold_proportion:
                idxs_to_remove.add(idx)

        # Delete the redundancies
        for idx in sorted(idxs_to_remove, reverse=True):
            pruned_boxes_xyxy = np.concatenate((pruned_boxes_xyxy[:idx], pruned_boxes_xyxy[idx + 1:]))
            pruned_logits = np.concatenate((pruned_logits[:idx], pruned_logits[idx + 1:]))
            del pruned_masks[idx]
            if verbose:
                print(f"below surface remove idx:{idx} phrase:{phrases[idx]}")
            del pruned_phrases[idx]

        return pruned_masks, pruned_boxes_xyxy, pruned_logits, pruned_phrases

    def disentangle_masks(self, masks, phrases, overlap_threshold_proportion=0.9, verbose=False):
        # If we have less than 2 masks, return immediately
        if len(masks) < 2:
            return masks

        # Sort by mask size
        keep_masks = [np.ones_like(masks[0]) for _ in range(len(masks))]

        for i, (mask_a, phrase_a) in enumerate(zip(masks, phrases)):
            assert mask_a.dtype == bool
            n_mask_pixels = mask_a.sum()
            for j, (mask_b, phrase_b) in enumerate(zip(masks, phrases)):
                # Skip self
                if i == j:
                    continue
                # Check if more than threshold amount of mask_a overlaps mask_b
                if (np.sum(mask_a & mask_b) / n_mask_pixels) > overlap_threshold_proportion:
                    # We will remove these pixels from mask_b
                    if verbose:
                        print(f"Mask from {phrase_a} is subsumed by mask from {phrase_b}. Removing mask pixels from {phrase_b}.")
                    keep_masks[j] &= ~mask_a

        # Update all masks
        updated_masks = [mask & keep_mask for mask, keep_mask in zip(masks, keep_masks)]

        return updated_masks
