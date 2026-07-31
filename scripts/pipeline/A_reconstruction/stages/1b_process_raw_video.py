# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from `simfoundry` env
"""
import shutil

import numpy as np
import os
from PIL import Image
import cv2
import time
from pathlib import Path
import hydra
from simfoundry import CFG_DIR
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage
import subprocess
import logging

logger = logging.getLogger(__name__)

bootstrap_hydra_workdir(__file__)


def subsample_images(source_folder, dest_folder, num_samples):
    """
    Subsamples a specified number of images from a source folder and copies them
    to a new destination folder.

    Args:
        source_folder (str): The path to the folder containing the original images.
        dest_folder (str): The path to the folder where the sampled images will be copied.
        num_samples (int): The number of images to randomly sample.
    """

    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created destination folder: {dest_folder}")

    # Get a list of all files in the source folder
    all_files = os.listdir(source_folder)

    # Filter for image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    image_files = list(sorted([f for f in all_files if f.lower().endswith(image_extensions)]))

    if len(image_files) == 0:
        raise ValueError(f"No image files found in source folder: {source_folder}")

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got: {num_samples}")

    # Check if there are enough images to sample
    if len(image_files) < num_samples:
        print(f"Warning: The source folder contains only {len(image_files)} images, "
              f"which is less than the requested {num_samples} samples.")
        num_samples = len(image_files)

    # Randomly select a subset of images
    skip_every = max(1, len(image_files) // num_samples)
    sampled_images = image_files[::skip_every][:num_samples]
    # sampled_images = random.sample(image_files, num_samples)

    # Copy the sampled images to the new folder
    for image_name in sampled_images:
        source_path = os.path.join(source_folder, image_name)
        dest_path = os.path.join(dest_folder, image_name)
        shutil.copy(source_path, dest_path)
        print(f"Copied {image_name}")

    print("\nImage subsampling complete!")


def process_single_image(input_image_fpath, out_video_dir, frames_all_dir):
    """
    Standardizes a single input image into the same directory layout expected by downstream stages.

    Args:
        input_image_fpath (str): Path to the input image.
        out_video_dir (str): Directory where the copied raw image and synthetic scene.mp4 will be written.
        frames_all_dir (str): Directory where frame_0001.png will be written.
    """
    input_image_path = Path(input_image_fpath)
    if not input_image_path.is_file():
        raise FileNotFoundError(f"Input image file does not exist: {input_image_fpath}")

    raw_image_fpath = f"{out_video_dir}/{input_image_path.name}"
    processed_video_fpath = f"{out_video_dir}/scene.mp4"
    frame_fpath = f"{frames_all_dir}/frame_0001.png"

    shutil.copy2(src=input_image_fpath, dst=raw_image_fpath)

    # Always write the canonical extracted frame as PNG for downstream stages.
    Image.open(input_image_fpath).convert("RGB").save(frame_fpath)

    # Synthesize a minimal video so the stage output remains structurally consistent.
    subprocess.run([
        "ffmpeg",
        "-y",
        "-loop", "1",
        "-i", raw_image_fpath,
        "-t", "1",
        "-r", "1",
        "-pix_fmt", "yuv420p",
        processed_video_fpath,
    ], check=True)

    return raw_image_fpath, processed_video_fpath


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    # Grab I/O info
    out_dir = cfg.s1_video.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Write input media to output path
    out_video_dir = f"{out_dir}/video"
    Path(out_video_dir).mkdir(parents=True, exist_ok=True)
    processed_video_fpath = f"{out_video_dir}/scene.mp4"
    frames_all_dir = f"{out_dir}/frames_all"
    Path(frames_all_dir).mkdir(parents=True, exist_ok=True)

    if cfg.s1_video.single_image_input:
        logger.info("="*60)
        logger.info("Processing single input image...")
        logger.info("="*60)
        process_single_image(
            input_image_fpath=cfg.s1_video.video_fpath.replace(".MOV", ".png"),
            out_video_dir=out_video_dir,
            frames_all_dir=frames_all_dir,
        )
    else:
        raw_video_fpath = f"{out_video_dir}/{os.path.basename(cfg.s1_video.video_fpath)}"
        shutil.copy2(src=cfg.s1_video.video_fpath, dst=raw_video_fpath)

        # Run ffmpeg to convert to standardized naming scheme format
        logger.info("="*60)
        logger.info("Copying input video file...")
        logger.info("="*60)
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", raw_video_fpath,
            processed_video_fpath,
        ], check=True)

        logger.info("="*60)
        logger.info("Extracting frames...")
        logger.info("="*60)
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", processed_video_fpath,
            f"{frames_all_dir}/frame_%04d.png",
        ], check=True)

    # Subsample frames
    n_subsampled_frames = cfg.s1_video.n_subsampled_frames
    frames_subsampled_dir = f"{out_dir}/frames_subsampled_{n_subsampled_frames}"
    Path(frames_subsampled_dir).mkdir(parents=True, exist_ok=True)
    subsample_images(source_folder=frames_all_dir, dest_folder=frames_subsampled_dir, num_samples=n_subsampled_frames)

    if cfg.s1_video.splat_prep:
        target_w = cfg.s1_video.target_w
        target_h = cfg.s1_video.target_h

        if target_w is not None and target_h is not None:
            logger.info("="*60)
            logger.info(f"splat_prep: resizing subsampled frames to {target_w}x{target_h}...")
            logger.info("="*60)
            for frame_name in sorted(os.listdir(frames_subsampled_dir)):
                if not frame_name.lower().endswith('.png'):
                    continue
                frame_path = os.path.join(frames_subsampled_dir, frame_name)
                img = Image.open(frame_path).convert("RGB")
                img.resize((target_w, target_h), Image.LANCZOS).save(frame_path)

        # Build input_video.mp4 from the subsampled PNGs (not from scene.mp4) so
        # the video frame count equals len(frames_subsampled_N) exactly.  SAM2
        # asserts that the video and the PNG directory have the same frame count,
        # so they must be in 1-to-1 correspondence.
        logger.info("="*60)
        logger.info("splat_prep: creating input_video.mp4 from subsampled frames...")
        logger.info("="*60)
        input_video_fpath = f"{out_dir}/input_video.mp4"
        # Write a sorted frame list file so ffmpeg reads them in the right order
        # regardless of filename gaps (frame_0001, frame_0016, ...).
        frame_list_fpath = f"{out_dir}/_splat_frame_list.txt"
        sorted_frames = sorted(
            f for f in os.listdir(frames_subsampled_dir) if f.lower().endswith('.png')
        )
        with open(frame_list_fpath, "w") as fl:
            for fname in sorted_frames:
                fl.write(f"file '{os.path.join(frames_subsampled_dir, fname)}'\n")
                fl.write(f"duration {1.0 / cfg.s1_video.mp4_fps}\n")
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", frame_list_fpath,
            "-pix_fmt", "yuv420p",
            input_video_fpath,
        ], check=True)

    finalize_stage(
        stage_cfg=cfg.s1_video,
        out_dir=cfg.s1_video.out_dir,
        result=StageResult(success=True),
    )


if __name__ == "__main__":
    main()
