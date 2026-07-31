# Copyright (c) 2024 the ACDC authors (Stanford Vision and Learning Lab)
# Licensed under the Apache License, Version 2.0.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file is derived from the ACDC / digital-cousins project
# (https://github.com/cremebrule/digital-cousins). NVIDIA modifications are licensed
# under Apache-2.0; the adapted upstream portions remain subject to the terms above.

import numpy as np
import torch
import torchvision
from torchvision.transforms import Resize, CenterCrop, InterpolationMode, Normalize
from PIL import Image
import clip
from simfoundry.models.visual_encoder import VisualEncoder
from simfoundry.models.text_encoder import TextEncoder


class CLIPEncoder(VisualEncoder, TextEncoder):
    EMBEDDING_DIMS = {
        "RN50": 1024,
        "RN101": 512,
        "RN50x4": 640,
        "RN50x16": 768,
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768,
    }

    def __init__(
            self,
            backbone_name="ViT-B/16",
            batch_size=None,
            preprocess_batch_size=128,
            device="cuda",
    ):
        """
        Args:
            backbone_name (str): Name of the backbone model. Valid options are CLIPEncoder.EMBEDDING_DIMS.keys()
            batch_size (None or int): If specified, batch size to use when computing features for a given set of images.
                This is used to avoid OOM errors with limited VRAM
            preprocess_batch_size (None or int): If specified, batch size to use when preprocessing images before
                passing them into the encoder. This is used to avoid OOM errors with limited VRAM
            device (str): device to store tensors on. Default is "cuda"
        """
        super().__init__(
            batch_size=batch_size,
            preprocess_batch_size=preprocess_batch_size,
            device=device,
        )

        # Sanity check backbone name
        assert backbone_name in self.EMBEDDING_DIMS.keys(), \
            f"Got invalid clip backbone name: {backbone_name}. Valid options are: {self.EMBEDDING_DIMS.keys()}"
        self.backbone_name = backbone_name

        self.backbone, _ = clip.load(backbone_name, device=device)
        self.backbone.eval()
        self.backbone.to(self.device)

    @property
    def embedding_dim(self):
        return self.EMBEDDING_DIMS[self.backbone_name]

    def preprocess(self, x):
        # Standardize shape to be 4-dim (B, H, W, C)
        if len(x.shape) < 4:
            x = x.reshape(1, *x.shape)

        # Convert from range [0, 255] -> [0.0, 1.0]
        x = torch.tensor(x, requires_grad=False) / 255.0

        # Permute, resize, normalize
        return torchvision.transforms.Compose(
            [
                torchvision.ops.Permute([0, 3, 1, 2]),
                Resize(self.backbone.visual.input_resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(self.backbone.visual.input_resolution),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )(x)

    def forward(self, x):
        return self.backbone.encode_image(x)

    def get_text_features(self, text):
        """
        Gets text features for given text @text

        Args:
            text (list of str): Text to encode

        Returns:
            np.ndarray: (N, D)-shaped array of text features
        """
        tokens = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.backbone.encode_text(tokens).cpu().numpy()

        return text_features

    def get_similarity_matrix(self, text_features):
        # Return cosine similarity matrix
        # TODO: Vectorize using faiss
        raise NotImplementedError
