# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sentence_transformers import SentenceTransformer
from simfoundry.models.text_encoder import TextEncoder


class SBERTEncoder(TextEncoder):
    BACKBONES = {
        "all-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1",
        "all-distilroberta-v1",
        "all-MiniLM-L6-v2",
    }

    def __init__(
            self,
            backbone_name="all-mpnet-base-v2",
            device="cuda",
    ):
        """
        Args:
            backbone_name (str): Name of the backbone model. Valid options are SBERTEncoder.BACKBONES
            device (str): device to store tensors on. Default is "cuda"
        """
        super().__init__()

        # Sanity check backbone name
        assert backbone_name in self.BACKBONES, \
            f"Got invalid clip backbone name: {backbone_name}. Valid options are: {self.BACKBONES}"
        self.backbone_name = backbone_name

        self.device = device
        self.backbone = SentenceTransformer(self.backbone_name)
        self.backbone.eval()
        self.backbone.to(self.device)

    def get_text_features(self, text):
        return self.backbone.encode(text)

    def get_similarity_matrix(self, text_features_a, text_features_b):
        return self.backbone.similarity(text_features_a, text_features_b)
