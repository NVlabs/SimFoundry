# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Adapted from https://github.com/arhanjain/sim-evals/blob/main/src/inference/droid_jointpos.py
"""

import numpy as np
from PIL import Image
from openpi_client import websocket_client_policy, image_tools
from simfoundry.utils.processing_utils import resize_with_pad

from .abstract_client import InferenceClient

class OpenPIClient(InferenceClient):
    def __init__(self, 
                host:str = "localhost", 
                port:int = 8000,
                open_loop_horizon:int = 8,
                 ) -> None:
        self.open_loop_horizon = open_loop_horizon
        print(f"Initializing OpenPI client with host: {host} and port: {port}")
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host, port
        )
        print(f"OpenPI client initialized")


    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        base_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        combined = np.concatenate([base_img, wrist_img], axis=1)
        return combined

    def reset(self):
        pass

    def infer(self, obs: dict, instruction: str) -> dict:
        """
        Infer the next action from the policy in a server-client setup
        """

        # TODO: remove this once its verify that image tools does not affect performance
        # curr_obs = self._extract_observation(obs)
        # request_data = {
        #     "observation/exterior_image_1_left": image_tools.resize_with_pad(
        #         obs["exterior_image_1_left"], 224, 224
        #     ),
        #     "observation/exterior_image_2_left": image_tools.resize_with_pad(
        #         obs["exterior_image_2_left"], 224, 224
        #     ),
        #     "observation/wrist_image_left": image_tools.resize_with_pad(
        #         obs["wrist_image_left"], 224, 224
        #     ),
        #     "observation/joint_position": obs["joint_position"],
        #     "observation/gripper_position": obs["gripper_position"],
        #     "prompt": instruction,
        # }

        ext_1_image = resize_with_pad(obs["exterior_image_1_left"], 224, 224)
        ext_2_image = resize_with_pad(obs["exterior_image_2_left"], 224, 224)
        wrist_image = resize_with_pad(obs["wrist_image_left"], 224, 224)


        request_data = {
            "observation/exterior_image_1_left": ext_1_image,
            "observation/exterior_image_2_left": ext_2_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": obs["joint_position"],
            "observation/gripper_position": obs["gripper_position"],
            "prompt": instruction,
        }
        vis_images = np.concatenate([ext_1_image, wrist_image], axis=1)
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

