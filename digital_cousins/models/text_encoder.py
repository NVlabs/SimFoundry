# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np


class TextEncoder(torch.nn.Module):
    """
    General class for encoding text features. Can have different backends, e.g.: CLIP, DINOv2, etc.
    """

    def get_text_features(self, text):
        """
        Gets text features for given text @text

        Args:
            text (list of str): Text to encode

        Returns:
            np.ndarray: (N, D)-shaped array of text features
        """
        raise NotImplementedError

    def get_similarity_matrix(self, text_features_a, text_features_b):
        raise NotImplementedError
