# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from PIL import Image
from openpi_client import websocket_client_policy, image_tools
from simfoundry.utils.processing_utils import resize_with_pad
import time
from .abstract_client import InferenceClient


class DreamZeroClient(InferenceClient):
    def __init__(self, 
                host:str = "localhost", 
                port:int = 8000,
                open_loop_horizon:int = 24,
                num_retries:int = 5,
                 ) -> None:
        self.open_loop_horizon = open_loop_horizon
        self.num_retries = num_retries
        print(f"Initializing DreamZero client with host: {host} and port: {port}")
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host, port
        )
        print(f"DreamZero client initialized")

    def reset(self):
        pass

    def infer(self, obs: dict, instruction: str) -> dict:
        """
        Infer the next action from the policy in a server-client setup
        """


        ext_1_image = resize_with_pad(obs["exterior_image_1_left"], 180, 320)
        ext_2_image = resize_with_pad(obs["exterior_image_2_left"], 180, 320)
        wrist_image = resize_with_pad(obs["wrist_image_left"], 180, 320)


        request_data = {
            "observation/exterior_image_1_left": ext_1_image,
            "observation/exterior_image_2_left": ext_2_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": obs["joint_position"],
            "observation/gripper_position": obs["gripper_position"],
            "session_id": "12203132026",
            "prompt": instruction,
            "endpoint": "infer",
        }
        vis_images = np.concatenate([ext_1_image, wrist_image], axis=1)

        for i in range(self.num_retries):
            try:
                pred_action_chunk = self.client.infer(request_data)["actions"]
                break
            except Exception as e:
                print(f"Error inferring action: {e}")
                time.sleep(1)

        pred_action_chunk = self.client.infer(request_data)["actions"]

        return {"action": pred_action_chunk, "viz": vis_images}



    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        # Assign images
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

