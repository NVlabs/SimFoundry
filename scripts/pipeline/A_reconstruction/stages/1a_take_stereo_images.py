# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from `zed` env

Requires installing:

- pyzed, see https://www.stereolabs.com/docs/development/python/install
"""

import pyzed.sl as sl
import numpy as np
import os
from PIL import Image
import cv2
import time
from pathlib import Path
import hydra
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage

from simfoundry import CFG_DIR

bootstrap_hydra_workdir(__file__)


def countdown_timer(seconds=3, message="Taking image in"):
    """
    Terminal-based countdown timer with tenths of seconds using tqdm
    
    Args:
        seconds (int): Number of seconds to count down
        message (str): Message to display before countdown
    """
    try:
        from tqdm import tqdm
        import sys
        
        print(f"{message}...")
        
        # Create progress bar that counts down
        total_tenths = seconds * 10
        with tqdm(total=total_tenths, desc="Countdown", 
                  bar_format="{desc}: {remaining}s |{bar}| {percentage:3.0f}%",
                  file=sys.stdout, leave=False) as pbar:
            
            for tenth in range(total_tenths, 0, -1):
                remaining_time = tenth / 10.0
                pbar.set_description(f"📷 {message} {remaining_time:.1f}s")
                pbar.update(1)
                time.sleep(0.1)
        
        print("📸 CAPTURE!")
        
    except ImportError:
        # Fallback to simple countdown if tqdm not available
        print(f"{message}...")
        for i in range(seconds, 0, -1):
            for tenth in range(10, 0, -1):
                remaining = i - 1 + tenth/10.0
                print(f"\r{remaining:.1f}s...", end="", flush=True)
                time.sleep(0.1)
        print("\r📸 CAPTURE!")


def countdown_timer_pygame(seconds=3, message="Taking image in", window_size=(400, 200)):
    """
    Pygame-based countdown timer with visual display
    
    Args:
        seconds (int): Number of seconds to count down
        message (str): Message to display before countdown
        window_size (tuple): Size of the countdown window (width, height)
    
    Note: Requires pygame to be installed: pip install pygame
    """
    try:
        import pygame
        import sys
        
        pygame.init()
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Image Capture Countdown")
        
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        
        # Fonts
        font_large = pygame.font.Font(None, 72)
        font_small = pygame.font.Font(None, 36)
        
        clock = pygame.time.Clock()
        
        # Count down with tenths of seconds precision
        start_time = time.time()
        total_time = seconds
        
        while True:
            elapsed = time.time() - start_time
            remaining = total_time - elapsed
            
            if remaining <= 0:
                break
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            screen.fill(WHITE)
            
            # Draw message
            text_msg = font_small.render(message, True, BLACK)
            msg_rect = text_msg.get_rect(center=(window_size[0]//2, 50))
            screen.blit(text_msg, msg_rect)
            
            # Draw countdown number with tenths
            color = RED if remaining <= 3.0 else BLACK
            countdown_text = f"{remaining:.1f}"
            text_num = font_large.render(countdown_text, True, color)
            num_rect = text_num.get_rect(center=(window_size[0]//2, window_size[1]//2))
            screen.blit(text_num, num_rect)
            
            pygame.display.flip()
            clock.tick(60)  # 60 FPS for smooth updates
        
        # Show "CAPTURE!" message
        for _ in range(30):  # Show for half a second at 60 FPS
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            screen.fill(WHITE)
            text_capture = font_large.render("CAPTURE!", True, GREEN)
            capture_rect = text_capture.get_rect(center=(window_size[0]//2, window_size[1]//2))
            screen.blit(text_capture, capture_rect)
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        
    except ImportError:
        print("Pygame not installed. Falling back to terminal countdown...")
        countdown_timer(seconds, message)


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    print("Starting image capture...")
    zed = sl.Camera()
    print("Camera initialized")

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(sl.RESOLUTION.HD2K, camera_fps=15)
    # init_params = sl.InitParameters(sl.RESOLUTION.HD1080)
    init_params.sdk_verbose = 0

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error initializing camera: {err}")
        exit(1)

    # Get camera information (ZED serial number)
    zed_serial = zed.get_camera_information().serial_number
    print(f"Hello! This is my serial number: {zed_serial}")

    # Set auto params
    zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)

    # Compute intrinsic matrix
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    # Focal length of the left eye in pixels
    focal_left_x = calibration_params.left_cam.fx
    # First radial distortion coefficient
    k1 = calibration_params.left_cam.disto[0]
    # Translation between left and right eye on x-axis
    tx = calibration_params.stereo_transform.get_translation().get()[0]
    # Horizontal field of view of the left eye in degrees
    h_fov = calibration_params.left_cam.h_fov

    # Construct K as average between the left and right Ks
    Ks = dict()
    for side, params in zip(["left", "right"], [calibration_params.left_cam, calibration_params.right_cam]):
        Ks[side] = np.array([
            [params.fx, 0, params.cx],
            [0, params.fy, params.cy],
            [0, 0, 1],
        ])
    K = (Ks["left"] + Ks["right"]) / 2

    save_dir = cfg.s1_zed.out_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{save_dir}/K.npy", K)
    # Also store with translation as txt file
    K_str = " ".join(str(k) for k in K.flatten())
    intrinsic_str = f"{K_str}\n{tx / 1000}\n"
    with open(f"{save_dir}/intrinsic.txt", "w") as f:
        f.writelines([intrinsic_str])

    # Save images
    n_images = cfg.s1_zed.n_images
    n_sec_between_images = cfg.s1_zed.n_sec_between_images
    use_pygame_countdown = getattr(cfg.s1_zed, 'use_pygame_countdown', False)  # Default to terminal countdown
    show_video_stream = getattr(cfg.s1_zed, 'show_video_stream', True)  # Default to showing video
    
    input("Press Enter to start image capture...")

    print("Starting image capture...")
    
    # Create OpenCV window if video stream is enabled
    if show_video_stream:
        window_name = "ZED Camera Stream (Press 'q' to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
    
    for i in range(n_images):
        # Countdown with live video feed
        start_time = time.time()
        countdown_duration = n_sec_between_images
        
        print(f"\nCapturing image {i + 1}/{n_images} in {countdown_duration} seconds...")
        
        while time.time() - start_time < countdown_duration:
            remaining = countdown_duration - (time.time() - start_time)
            
            # Grab frame for video display
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                if show_video_stream:
                    # Get left camera view for display
                    image = sl.Mat()
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    frame = image.get_data()[:, :, :3]
                    
                    # Add countdown text overlay
                    frame_display = frame.copy()
                    text = f"Capturing {i + 1}/{n_images} in {remaining:.1f}s"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness = 3
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    text_x = (frame_display.shape[1] - text_size[0]) // 2
                    text_y = 60
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame_display, 
                                (text_x - 10, text_y - text_size[1] - 10),
                                (text_x + text_size[0] + 10, text_y + 10),
                                (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(frame_display, text, (text_x, text_y),
                              font, font_scale, (0, 255, 0), thickness)
                    
                    # Display frame
                    cv2.imshow(window_name, frame_display)
                    
                    # Check for 'q' key to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nCapture cancelled by user")
                        cv2.destroyAllWindows()
                        zed.close()
                        return
            
            time.sleep(0.033)  # ~30 FPS update rate
        
        # Capture final image
        zed.grab()
        print(f"✓ Captured image {i + 1} / {n_images}")
        
        for side, view in zip(["left", "right"], [sl.VIEW.LEFT, sl.VIEW.RIGHT]):
            image = sl.Mat()
            zed.retrieve_image(image, view)
            Image.fromarray(image.get_data()[:, :, :3][:, :, ::-1]).save(f"{save_dir}/zed_{i}_{side[0]}.png")
        
        # Brief flash to indicate capture
        if show_video_stream and i < n_images - 1:  # Don't flash after last image
            image = sl.Mat()
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()[:, :, :3]
            white_flash = np.ones_like(frame) * 255
            cv2.imshow(window_name, white_flash.astype(np.uint8))
            cv2.waitKey(100)

    # Cleanup
    if show_video_stream:
        cv2.destroyAllWindows()
    
    zed.close()
    
    finalize_stage(
        stage_cfg=cfg.s1_zed,
        out_dir=cfg.s1_zed.out_dir,
        result=StageResult(success=True),
    )


if __name__ == "__main__":
    main()
