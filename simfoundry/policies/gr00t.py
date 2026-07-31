# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from .abstract_client import InferenceClient
import io
import numpy as np
import torch
from .gr00t_utils import PolicyClient
from simfoundry.utils.processing_utils import resize_with_pad
from PIL import Image
from scipy.spatial.transform import Rotation

RESOLUTION = (180, 320)

DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    dtype=np.float64,
)


def _rot_mat_to_rot_6d(rot_mat: np.ndarray) -> np.ndarray:
    """Extract first two rows of a 3x3 rotation matrix as rot6d (6,)."""
    return rot_mat[:2, :].flatten().astype(np.float32)


def _quat_to_eef_9d(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert EEF pos (3,) + quaternion xyzw (4,) to eef_9d (9,): xyz + rot6d."""
    rot_mat = Rotation.from_quat(quat_xyzw).as_matrix()
    rot_mat_corrected = rot_mat @ DROID_EEF_ROTATION_CORRECT
    rot6d = _rot_mat_to_rot_6d(rot_mat_corrected)
    return np.concatenate([pos.astype(np.float32), rot6d])


def _quat_to_cartesian(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert EEF pos (3,) + quaternion xyzw (4,) to cartesian_position (6,): xyz + euler_xyz."""
    euler = Rotation.from_quat(quat_xyzw).as_euler("xyz")
    return np.concatenate([pos, euler]).astype(np.float64)


def _jpeg_encode(image: np.ndarray, quality: int = 95) -> bytes:
    """JPEG-encode a uint8 (H, W, C) image and return raw bytes."""
    img = Image.fromarray(image.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _detect_policy_mode(modality_config: dict) -> str:
    """Detect policy version from server modality config.

    Returns 'n17' for oxe_droid_relative_eef_relative_joint,
    'n16' for oxe_droid_joint_position_relative (Gr00t),
    or 'generic' as fallback.
    """
    if "state" in modality_config and "video" in modality_config:
        state_cfg = modality_config["state"]
        skeys = set(state_cfg.modality_keys)
        if "eef_9d" in skeys:
            return "n17"
    if "video" in modality_config:
        video_cfg = modality_config["video"]
        deltas = list(video_cfg.delta_indices)
        if len(deltas) == 1 and int(deltas[0]) == 0:
            return "n16"
    return "generic"


class Gr00tClient(InferenceClient):
    def __init__(self, host: str = "localhost", port: int = 5555, api_token: str = None, open_loop_horizon: int = 12):
        print(f"Initializing Gr00t client with host: {host} and port: {port}")
        self.client = PolicyClient(host=host, port=port, api_token=api_token)
        print(f"Gr00t client initialized")
        self.modality_config = self.client.get_modality_config()
        self.open_loop_horizon = open_loop_horizon
        self.policy_mode = _detect_policy_mode(self.modality_config)
        print(f"Detected policy mode: {self.policy_mode}")


    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        base_img = resize_with_pad(curr_obs["right_image"], RESOLUTION[0], RESOLUTION[1])
        wrist_img = resize_with_pad(curr_obs["wrist_image"], RESOLUTION[0], RESOLUTION[1])
        combined = np.concatenate([base_img, wrist_img], axis=1)
        return combined

    def infer(self, obs: dict, instruction: str) -> dict:
        if self.policy_mode == "n17":
            return self._infer_n17(obs, instruction)
        return self._infer_n16(obs, instruction)

    def _infer_n16(self, obs: dict, instruction: str) -> dict:
        """N1.6 (Gr00t) request: raw numpy images, batch-dim actions."""
        request_data = {}

        if 'state' in self.modality_config:
            for key in self.modality_config['state'].modality_keys:
                if key in obs:
                    request_data[f'state.{key}'] = obs[key][None, None, ...].astype(np.float32)

        vis_images = []
        for key in self.modality_config['video'].modality_keys:
            resized = resize_with_pad(obs[key], RESOLUTION[0], RESOLUTION[1])
            request_data[f'video.{key}'] = resized[None, None, ...]
            vis_images.append(resized)

        for key in self.modality_config['language'].modality_keys:
            request_data[key] = [instruction]

        response = self.client.get_action(request_data)

        action_concat = []
        for key in self.modality_config['action'].modality_keys:
            action_concat.append(response[0][f'action.{key}'][0])
        pred_action_chunk = np.concatenate(action_concat, axis=1)

        vis_images = np.concatenate(vis_images, axis=1)
        return {"action": pred_action_chunk, "viz": vis_images}

    def _infer_n17(self, obs: dict, instruction: str) -> dict:
        """N1.7 request: JPEG-encoded images, no extra batch dim on actions.

        Matches N17eefRelative.build_request() from droid_control_loop.
        The N1.7 server (Gr00tG1RealPolicyWrapper) JPEG-decodes video
        and has two language-key lookup paths, so we send both variants.
        """
        request_data = {}

        if 'state' in self.modality_config:
            for key in self.modality_config['state'].modality_keys:
                if key in obs:
                    request_data[f'state.{key}'] = obs[key][None, None, ...].astype(np.float32)

        vis_images = []
        for key in self.modality_config['video'].modality_keys:
            resized = resize_with_pad(obs[key], RESOLUTION[0], RESOLUTION[1])
            vis_images.append(resized)
            request_data[f'video.{key}'] = [[_jpeg_encode(resized)]]

        lang = [instruction]
        for key in self.modality_config['language'].modality_keys:
            request_data[key] = lang
            alt_key = key.replace("annotation.language.", "annotation.")
            if alt_key != key:
                request_data[alt_key] = lang

        response = self.client.get_action(request_data)

        # N1.7 returns eef_9d + gripper_position + joint_position, but only
        # joint_position + gripper_position are used for execution (matching
        # the DROID control loop which ignores the relative eef_9d action).
        joint_actions = response[0]["action.joint_position"]
        gripper_actions = response[0]["action.gripper_position"]
        min_horizon = min(joint_actions.shape[0], gripper_actions.shape[0])
        pred_action_chunk = np.concatenate(
            (joint_actions[:min_horizon], gripper_actions[:min_horizon]),
            axis=1,
        )

        vis_images = np.concatenate(vis_images, axis=1)
        return {"action": pred_action_chunk, "viz": vis_images}

    def reset(self):
        self.client.reset()


    def _extract_observation(self, obs_dict, *, robot=None, save_to_disk=False):
        robot_state = obs_dict["policy"]

        # Images — left defaults to right (single external cam in sim)
        right_image = robot_state["external_cam"][0].clone().detach().cpu().numpy()
        if "external_cam_2" in robot_state:
            left_image = robot_state["external_cam_2"][0].clone().detach().cpu().numpy()
        else:
            left_image = right_image
        wrist_image = robot_state["wrist_cam"][0].clone().detach().cpu().numpy()

        # Proprioceptive state
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        # EEF pose at panda_link8 (matches real-DROID Pinocchio FK frame).
        # Requires the OmniGibson robot object to read panda_link7 and apply
        # the +107 mm Z offset to replicate panda_link8 / panda_joint8.
        cartesian_position = None
        eef_9d = None
        if robot is not None and "panda_link7" in robot.links:
            link7 = robot.links["panda_link7"]
            pos7, quat7 = link7.get_position_orientation()
            pos7_np = pos7.cpu().numpy().astype(np.float32)
            quat7_np = quat7.cpu().numpy().astype(np.float32)  # xyzw
            z_world = Rotation.from_quat(quat7_np).apply(np.array([0.0, 0.0, 0.107]))
            eef_pos = (pos7_np + z_world).astype(np.float32)
            eef_quat = quat7_np
            cartesian_position = _quat_to_cartesian(eef_pos, eef_quat)
            eef_9d = _quat_to_eef_9d(eef_pos, eef_quat)

        if save_to_disk:
            combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        result = {
            "left_image": left_image,
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
        if cartesian_position is not None:
            result["cartesian_position"] = cartesian_position
            result["eef_position"] = cartesian_position[:3].astype(np.float32)
            result["eef_rotation"] = cartesian_position[3:].astype(np.float32)
        if eef_9d is not None:
            result["eef_9d"] = eef_9d
        return result