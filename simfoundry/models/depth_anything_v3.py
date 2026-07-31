# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import shutil
from typing import Union, List

import simfoundry
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

from simfoundry.utils.processing_utils import process_depth_linear, unprocess_depth_linear


def _wait_for_complete_npz(npz_path, timeout=600.0, poll=1.0):
    """Block until @npz_path is a complete, loadable .npz, or @timeout seconds elapse.

    DA3 exports ``results.npz`` from a fire-and-forget background thread
    (``depth_anything_3.utils.parallel_utils.async_call`` -> ``np.savez_compressed``),
    so ``model.inference()`` returns BEFORE the file is fully written — a caller that
    immediately checks/loads the npz races the writer. We poll until the npz's zip
    central directory (written last) is readable, which means the write has finished.
    Returns True if the npz completed, False on timeout.
    """
    npz_path = Path(npz_path)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if npz_path.exists():
            try:
                with np.load(npz_path) as d:
                    _ = d.files  # forces reading the zip central directory
                return True
            except (zipfile.BadZipFile, EOFError, ValueError, OSError):
                pass  # still being written
        time.sleep(poll)
    return False


class DepthAnythingV3(torch.nn.Module):
    """
    Thin wrapper around DepthAnything V3 model for inferring depth maps
    """
    # Supported configurations to use
    # See https://github.com/ByteDance-Seed/Depth-Anything-3?tab=readme-ov-file#%EF%B8%8F-model-cards for valid options
    CONFIGS = {
        "DA3NESTED-GIANT-LARGE",
        "DA3NESTED-GIANT-LARGE-1.1",
    }

    def __init__(
            self,
            model_name="DA3NESTED-GIANT-LARGE-1.1",
            device="cuda",
    ):
        """
        Args:
            model_name (str): Size of underlying ViT model to use. Options are {"DA3NESTED-GIANT-LARGE"}
            max_depth (None or int): Maximum depth (m) to use for depth estimation. If None, will use a default
                based on @backbone_dataset
            device (str): device to store tensors on. Default is "cuda"
        """
        # Call super first
        super().__init__()

        # Sanity check values
        assert model_name in self.CONFIGS,\
            f"Got invalid DepthAnythingV3 model_name! Valid options: {self.CONFIGS}, got: {model_name}"
        self.device = device

        # Load model
        self.model = DepthAnything3.from_pretrained(f"depth-anything/{model_name}")
        self.model = self.model.to(self.device)
        self.model.eval()

    def infer_depth(
            self,
            image: Union[np.ndarray, Image.Image, str, List],
            resolution: int = 504,
            resize_to_input: bool = True,
    ) -> np.ndarray:
        """
        Runs depth inference on one or more images and returns depth maps directly (no disk I/O).

        Args:
            image: Input image(s). Can be a single numpy array (H, W, 3), PIL Image, file path string,
                or a list of any of these types.
            resolution (int): Processing resolution for max(H, W). Must be a multiple of 14.
                Default is 504.
            resize_to_input (bool): If True, resize the output depth map(s) to match the spatial
                dimensions of the input image(s). Default is True.

        Returns:
            np.ndarray: Depth map with shape (H, W) for a single image, or (N, H, W) for multiple
                images. When resize_to_input is True, (H, W) matches the input image dimensions;
                otherwise (H, W) corresponds to the model's processed resolution.
        """
        # Normalize to list
        single_image = not isinstance(image, list)
        images = [image] if single_image else image

        # Resolve input dimensions for optional resizing
        input_sizes = []
        if resize_to_input:
            for img in images:
                if isinstance(img, np.ndarray):
                    input_sizes.append((img.shape[0], img.shape[1]))
                elif isinstance(img, Image.Image):
                    w, h = img.size
                    input_sizes.append((h, w))
                elif isinstance(img, str):
                    pil_img = Image.open(img)
                    w, h = pil_img.size
                    input_sizes.append((h, w))
                else:
                    raise TypeError(f"Unsupported image type: {type(img)}")

        prediction = self.model.inference(
            image=images,
            process_res=resolution,
            process_res_method="upper_bound_resize",
            export_dir=None,
            export_format="npz",
        )
        depth = prediction.depth  # (N, H_proc, W_proc)

        if resize_to_input:
            resized = []
            for i, d in enumerate(depth):
                target_h, target_w = input_sizes[i]
                if d.shape != (target_h, target_w):
                    d = cv2.resize(d, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                resized.append(d)
            depth = np.stack(resized, axis=0)

        if single_image:
            return depth[0]
        return depth

    def infer(
            self,
            image_fpaths,
            output_dir,
            resolution,
            extrinsics=None,
            intrinsics=None,
            include_mesh=False,
            include_gs=False,
            include_gs_video=False,
            include_colmap=False,
    ):
        """
        Runs inference on batch of images using DA3 model. All results are written to @output_dir

        Args:
            image_fpaths (list of str): List of absolute image paths to run inference on
            output_dir (str): Absolute path to output directory where results will be stored
            resolution (int): Resolution to use for max(H, W) when resizing images before passing them through model.
                Must be multiple of 14
            extrinsics (None or np.ndarray): If specified, known extrinsics to use for images (may improve output fidelity)
            intrinsics (None or np.ndarray): If specified, known intrinsics to use for images (may improve output fidelity)
            include_mesh (bool): Whether to include mesh as part of output
            include_gs (bool): Whether to include gaussian splat as part of output
            include_gs_video (bool): Whether to include rendered video of gaussian splat as part of output
            include_colmap (bool): Whether to include camera info in COLMAP format as part of output
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        export_format = "npz"
        infer_gs = False
        if include_mesh:
            export_format += "-glb"
        if include_gs:
            export_format += "-gs_ply"
            infer_gs = True
        if include_gs_video:
            export_format += "-gs_video"
            infer_gs = True
        if include_colmap:
            export_format += "-colmap"

        prediction = self.model.inference(
            image=image_fpaths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            process_res=resolution,
            process_res_method="upper_bound_resize",
            export_dir=output_dir,
            export_format=export_format,
            align_to_input_ext_scale=True,
            infer_gs=infer_gs,  # Required for gs_ply and gs_video exports
        )

        # DA3 writes results.npz in a background thread (parallel_utils.async_call), so it
        # may be absent/partial when inference() returns. Wait for the complete file before
        # returning so callers (chunk worker, single-pass backend) don't race the writer.
        npz_path = Path(output_dir) / "exports" / "npz" / "results.npz"
        if not _wait_for_complete_npz(npz_path):
            raise RuntimeError(f"DA3 results.npz did not finish writing within timeout: {npz_path}")

        # If we wrote to colmap, unify in single directory
        if include_colmap:
            colmap_dir = f"{output_dir}/sparse"
            Path(colmap_dir).mkdir(parents=True, exist_ok=True)
            files = ["cameras.bin", "frames.bin", "images.bin", "points3D.bin", "rigs.bin"]
            for file in files:
                shutil.move(f"{output_dir}/{file}", f"{colmap_dir}/{file}")

            # We also copy all original images over in case we want to train a 3DGS using COLMAP values
            for img_fpath in image_fpaths:
                img_dir = f"{output_dir}/images"
                Path(img_dir).mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_fpath, f"{img_dir}/{os.path.basename(img_fpath)}")
