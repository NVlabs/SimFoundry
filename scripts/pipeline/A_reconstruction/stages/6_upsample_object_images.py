# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from `simfoundry` env

Requires installing:

- simfoundry, see the main README

Don't forget to register API key from openai / Gemini CLI!
"""
import cv2
import logging

from simfoundry.models.vlm import GPT, FLUX1, Gemini
from simfoundry.utils.prompt_utils import prompt_upsample_image, prompt_upsample_image_rotate, \
    prompt_flux_object_completion_and_upsample_preserve, prompt_upsample_image_gemini, prompt_infill_image, prompt_infill_image_no_conditioning, \
    prompt_check_object_validity, parse_json_response
from simfoundry.utils.processing_utils import pad_image_to_ratio, remove_background
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage
from simfoundry.utils.python_utils import assert_valid_key
import torch
from pathlib import Path
from PIL import Image
import json
import os 
import numpy as np
import hydra


# Set up logger
logger = logging.getLogger(__name__)

bootstrap_hydra_workdir(__file__)

REMBG_CPU_PROVIDERS = ["CPUExecutionProvider"]


UPSAMPLE_MODELS = {
    "gemini",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image",
    "gpt",
    "flux",
}

from simfoundry import CFG_DIR


def resolve_requested_indices(cfg):
    """Resolve optional per-object filter for streaming/partial reruns."""
    raw = cfg.s6_upsample.get("object_indices", None)
    if raw is None:
        return None
    if isinstance(raw, int):
        return {raw}
    return {int(v) for v in raw}


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    img_dir = cfg.s5_scene.out_dir + "/masked_object"
    out_dir = cfg.s6_upsample.out_dir
    
    dirs_to_create = ["padded", "infilled", "upsampled", "validity"]
    for dir_name in dirs_to_create:
        Path(f"{out_dir}/{dir_name}").mkdir(parents=True, exist_ok=True)
        
    logger.info("="*60)
    logger.info("Starting object upsampling pipeline")
    logger.info(f"Input directory: {img_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*60)

    rembg_session = None
    remove_bg_max_side = cfg.s6_upsample.get("remove_bg_max_side", None)

    # Create relevant client API to use
    # Calculate target ratio based on output image shape
    model_name = cfg.s6_upsample.model
    logger.info(f"Using upsample model: {model_name}")
    assert_valid_key(key=model_name, valid_keys=UPSAMPLE_MODELS, name="upsample model")
    ratios = dict()
    invalid_objects = []
    if model_name == "gemini":
        model = Gemini(
            project=cfg.gcloud_project,
            location="global",
            model="gemini-3-pro-image",
        )
        for (out_w, out_h) in model.IMAGE_SHAPES:
            ratio = out_w / out_h
            ratios[(out_w, out_h)] = {
                "ratio": ratio,
            }
        upsample_prompt = prompt_upsample_image_gemini()
    elif "gemini" in model_name:
        model = Gemini(
            project=cfg.gcloud_project,
            location="global",
            model=model_name,
        )
        for (out_w, out_h) in model.IMAGE_SHAPES:
            ratio = out_w / out_h
            ratios[(out_w, out_h)] = {
                "ratio": ratio,
            }
        upsample_prompt = prompt_upsample_image_gemini()
    elif model_name == "gpt":
        model = GPT(
            model="gpt-image-1",
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        for image_shape, resolution_str in model.IMAGE_SHAPES.items():
            out_w, out_h = [int(x) for x in resolution_str.split("x")]
            ratio = out_w / out_h
            ratios[(out_w, out_h)] = {
                "ratio": ratio,
                "image_shape": image_shape,
            }
        upsample_prompt = prompt_upsample_image_rotate()
    elif model_name == "flux":
        model = FLUX1(
            model="FLUX.1-Kontext-dev",
            dtype=torch.bfloat16,
            device="cuda",
        )
        for (out_w, out_h) in model.IMAGE_SHAPES:
            ratio = out_w / out_h
            ratios[(out_w, out_h)] = {
                "ratio": ratio,
            }
        # Flux prompt is object-specific; format it inside the per-object loop.
        upsample_prompt = None
        infill_prompt = None
    else:
        raise NotImplementedError

    try:
        from rembg import new_session

        rembg_session = new_session("bria-rmbg", providers=REMBG_CPU_PROVIDERS)
        logger.info("Initialized shared rembg session for stage 6 on CPU")
    except Exception as e:
        logger.warning(f"Failed to initialize shared rembg session, falling back to per-call session creation: {e}")

    # for checking object validity at the end of upsampling
    check_valid = cfg.s6_upsample.check_valid
    if check_valid:
        validity_model_name = cfg.s6_upsample.validity_model
        if validity_model_name == "gemini":
            validity_model = Gemini(
                project=cfg.gcloud_project,
                location="global",
                model="gemini-3-pro-preview",
            )
        else:
            raise NotImplementedError
    else:
        logger.info("Skipping validity check (check_valid=false)")


    def call_image_editing_model(model, model_name, obj_phrase, img_path, cfg, prompt, task="infill"):

        if prompt is None:
            logger.warning(f"Prompt is None for {task} task on {obj_phrase}")
            return Image.open(img_path), None

        logger.info(f"Using {model.__class__.__name__} to {task} [{obj_phrase}] at {img_path}...")
            
        if model_name == "gemini":
            assert isinstance(model, Gemini)
            result = model(
                prompt=prompt,
                image_paths=img_path,
                temperature=0,
                top_p=0,
                print_results=cfg.visualize,
            )
            if result is None:
                raise RuntimeError(f"Gemini returned no result for {task} on [{obj_phrase}] at {img_path}")
        elif "gemini" in model_name:
            assert isinstance(model, Gemini)
            result = model(
                prompt=prompt,
                image_paths=img_path,
                temperature=0,
                top_p=0,
                print_results=cfg.visualize,
            )
            if result is None:
                raise RuntimeError(f"Gemini returned no result for {task} on [{obj_phrase}] at {img_path}")
        elif model_name == "gpt":
            assert isinstance(model, GPT)
            result = model(
                prompt=prompt,
                image_path=img_path,
                n_images=1,
                n_retries=3,
                image_shape=ratios[target_wh]["image_shape"],
                print_results=cfg.visualize,
            )
            if result is None:
                raise RuntimeError(f"GPT returned no result for {task} on [{obj_phrase}] at {img_path}")
        elif model_name == "flux":
            assert isinstance(model, FLUX1)
            image = model(
                prompt=prompt,
                image_path=img_path,
                guidance_scale=2.5,
                num_inference_steps=20,
                max_sequence_length=512,
                print_results=cfg.visualize,
            )
            return image, image
        else:
            raise NotImplementedError

        return model.get_result_images(result=result)[0], result

    # out_w, out_h = [int(x) for x in gpt.IMAGE_SHAPES[cfg.s6_upsample.image_shape].split("x")]
    # target_ratio = out_w / out_h

    # Iterate over all files in the img dir, and pass them through GPT to upsample
    # TODO: Sorted filename is inconsistent because of the numbering system (e.g. 1 vs 12)
    # TODO: Blur the edge pixels to avoid some artifacts during generation
    all_files = sorted(os.listdir(img_dir))
    logger.info(f"Found {len([f for f in all_files if f.endswith('.png')])} image files to process")
    requested_indices = resolve_requested_indices(cfg)
    if requested_indices is not None:
        logger.info("Filtering to requested object indices: %s", sorted(requested_indices))
    
    for filename in all_files:
        if not filename.endswith(".png"):
            continue
        img_name = filename.split(".")[0]
        if "iter" not in filename:
            continue
        idx = int(img_name.split("iter_")[-1])
        if requested_indices is not None and idx not in requested_indices:
            continue

        # if idx != 1:
        #     continue

        logger.info(f"Processing {filename} (iteration {idx})")
        
        input_img_path = f"{img_dir}/{filename}"
        obj_cat_info_fpath = f"{img_dir}/../obj_cat_list/iter_{idx}.json"
        with open(obj_cat_info_fpath) as f:
            obj_cat_info = json.load(f)

        # Skip any invalid ones (unless include_invalid=true in config)
        include_invalid = cfg.s6_upsample.get("include_invalid", False)
        if not obj_cat_info["is_valid_removed_obj"]:
            if not include_invalid:
                logger.warning(f"Skipping {filename}: marked as invalid removed object (set s6_upsample.include_invalid=true to process anyway)")
                continue
            else:
                logger.warning(f"Processing {filename} despite being marked as invalid removed object (include_invalid=true)")

        obj_phrase = obj_cat_info["removed_obj_phrase"]
        logger.info(f"Processing object: {obj_phrase}")

        # Load image and prune / pad the image to the image ratio that maximally fills the image with the object
        # foreground before passing it to GPT
        # TODO: Don't maximally fill if not using GPT?
        input_img = np.array(Image.open(input_img_path))
        H, W, _ = input_img.shape
        mask = input_img[:, :, -1]
        mask_h_idxs, mask_w_idxs = np.where(mask)
        min_h, max_h = mask_h_idxs.min(), mask_h_idxs.max()
        min_w, max_w = mask_w_idxs.min(), mask_w_idxs.max()
        margin = int(np.ceil(max(max_h - min_h, max_w - min_w) * 0.1))
        img_min_ratio = (max_w - min_w) / (max_h - min_h)
        min_dist = np.inf
        target_wh = None
        for wh, info in ratios.items():
            dist = np.abs(img_min_ratio - info["ratio"])
            if dist < min_dist:
                min_dist = dist
                target_wh = wh
        # Crop image, then pad to desired ratio
        cropped_image = input_img[
        max(0, min_h - margin): min(max_h + 1 + margin, H),
        max(0, min_w - margin): min(max_w + 1 + margin, W),
        :
        ]
        input_img_padded = pad_image_to_ratio(cropped_image, target_ratio=ratios[target_wh]["ratio"], padding_color=(0, 0, 0))
        # If we're using flux, then resize to the desired resolution
        if model_name == "flux":
            input_img_padded = cv2.resize(input_img_padded, target_wh)
        input_img_padded_path = f"{out_dir}/padded/{img_name}.png"
        Image.fromarray(input_img_padded).save(input_img_padded_path)

        # # Load image and pad it to the appropriate ratio before passing it to GPT
        # input_img = np.array(Image.open(input_img_path))
        # input_img_padded = pad_image_to_ratio(input_img, target_ratio=target_ratio, padding_color=(0, 0, 0))
        # input_img_padded_path = f"{out_dir}/{img_name}_padded.png"
        # Image.fromarray(input_img_padded).save(input_img_padded_path)

        #########################################################################################################
        ########### INFILL STEP: complete the objects by infill before upsampling
        ###########
        #########################################################################################################
        if cfg.s6_upsample.infill:
            infill_prompt = prompt_infill_image_no_conditioning(obj_name=obj_phrase)
            infilled_img, result  = call_image_editing_model(model, model_name, obj_phrase, input_img_padded_path, cfg, infill_prompt, task="infill")

            infilled_img_fpath = f"{out_dir}/infilled/{img_name}.png"
            infilled_img.save(infilled_img_fpath)

            # Remove background from infilled image
            remove_background(
                infilled_img_fpath,
                infilled_img_fpath,
                session=rembg_session,
                max_side=remove_bg_max_side,
            )
        
        else:
            infilled_img_fpath = input_img_padded_path

        #########################################################################################################
        ########### UPSAMPLING STEP: upsample the object to the desired resolution
        ###########
        #########################################################################################################
        current_upsample_prompt = (
            prompt_flux_object_completion_and_upsample_preserve(obj_phrase=obj_phrase)
            if model_name == "flux"
            else upsample_prompt
        )
        upsampled_img, result  = call_image_editing_model(
            model, model_name, obj_phrase, infilled_img_fpath, cfg, current_upsample_prompt, task="upsample"
        )
        upsampled_img_fpath = f"{out_dir}/upsampled/{img_name}.png"
        upsampled_img.save(upsampled_img_fpath)

        # # Remove background from this image
        # pil_img = Image.open(upsampled_img_fpath)
        # np_img = np.array(pil_img)
        # masks, boxes_xyxy, logits = sam3.predict_segmentation(pil_img=pil_img, text_prompt=obj_phrase)
        # if len(masks) == 0:
        #     logger.warning(f"Found no foreground with phrase [{obj_phrase}] in the upsampled image!")
        #     continue
        # # all_masks = gsam.predict_segmentation(image_source, boxes_xyxy, multimask_output=False)
        # # success = True

        # # if not success:
        # #     logger.error(f"Failed to upsample object [{obj_phrase}] with valid foreground / background split!")
        # #     from IPython import embed; embed()

        # # from IPython import embed; embed()

        # # # Prune the masks
        # # pruned_masks, pruned_boxes_xyxy, pruned_logits, pruned_phrases = sam3.prune_redundant_masks(
        # #     masks=[mask.squeeze(axis=0) for mask in masks],
        # #     boxes_xyxy=boxes_xyxy,
        # #     logits=logits,
        # #     phrases=[obj_phrase] * len(masks),
        # #     polygon_relative_intersection_threshold=0.97,
        # #     polygon_relative_area_threshold=0.9,
        # #     obj_mask_intersect_area_threshold=0.8,
        # #     verbose=True,
        # # )

        # if len(masks) != 1:
        #     logger.warning(f"Got {len(masks)} masks when pruning foreground (expected 1). Using the union of these masks.")
        #     pruned_mask = np.any(np.stack(masks, axis=0), axis=0)[0]
        # else:
        #     pruned_mask = masks[0][0]

        # # Only keep largest (inverted) mask as the foreground object
        # contours, _ = cv2.findContours((pruned_mask).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # largest_contour = None
        # max_area = 0

        # for contour in contours:
        #     area = cv2.contourArea(contour)
        #     if area > max_area:
        #         max_area = area
        #         largest_contour = contour

        # foreground_mask = np.zeros_like(pruned_mask, dtype=np.uint8)
        # cv2.drawContours(foreground_mask, [largest_contour], -1, 255, cv2.FILLED)
        # image_source_masked = np.concatenate([np_img, foreground_mask[..., np.newaxis]], axis=-1)
        # Image.fromarray(image_source_masked).save(f"{out_dir}/upsampled/{img_name}_transparent.png")

        # Remove background
        remove_background(
            Path(upsampled_img_fpath),
            Path(f"{out_dir}/upsampled/{img_name}_transparent.png"),
            session=rembg_session,
            max_side=remove_bg_max_side,
        )

        
        #########################################################################################################
        ########### CHECK OBJECTS STEP: check if the objects are valid
        ###########
        #########################################################################################################

        if check_valid:
            result = validity_model(
                prompt=prompt_check_object_validity(obj_phrase),
                image_paths=upsampled_img_fpath,
            )
            result_text = validity_model.get_result_text(result=result)
            # Sometimes we get malformed jsons, so we try / except here
            try:
                validity_result = parse_json_response(result_text)
                with open(f"{out_dir}/validity/{img_name}.json", "w") as f:
                    json.dump(validity_result, f)
                valid = validity_result["validity"] == "yes"
            except Exception as e:
                logger.warning(f"Failed to validate, error: {e}")
                logger.warning(f"Directly setting valid=True instead!")
                valid = True
            if not valid:
                logger.warning(f"Object [{obj_phrase}] is not valid!")
                invalid_objects.append(obj_phrase) # Save here for later confirmation by human
            else:
                logger.info(f"Object [{obj_phrase}] is valid!")
        else:
            logger.info(f"Skipping validity check for [{obj_phrase}]")

    logger.info("="*50)
    logger.info(f"Upsampling complete!")
    logger.info(f"Total invalid objects: {len(invalid_objects)}")
    if invalid_objects:
        logger.warning(f"Invalid objects: {invalid_objects}")
    logger.info("="*50)
    
    finalize_stage(
        stage_cfg=cfg.s6_upsample,
        out_dir=cfg.s6_upsample.out_dir,
        result=StageResult(success=True, additional_info={"invalid_objects": invalid_objects}),
    )

if __name__ == "__main__":
    main()
