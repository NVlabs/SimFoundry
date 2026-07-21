import torch
import numpy as np


class VisualEncoder(torch.nn.Module):
    """
    General class for encoding visual features in ACDC. Can have different backends, e.g.: CLIP, DINOv2, etc.
    """
    def __init__(
            self,
            batch_size=None,
            preprocess_batch_size=None,
            device="cuda",
    ):
        """
        Args:
            batch_size (None or int): If specified, batch size to use when computing features for a given set of images.
                This is used to avoid OOM errors with limited VRAM
            preprocess_batch_size (None or int): If specified, batch size to use when preprocessing images before
                passing them into the encoder. This is used to avoid OOM errors with limited VRAM
            device (str): device to store tensors on. Default is "cuda"
        """
        # Always run super first
        super().__init__()

        # Store internal variables
        self.batch_size = batch_size
        self.preprocess_batch_size = preprocess_batch_size
        self.device = device

    def preprocess(self, x):
        """
        Preprocess raw RGB inputs @x and converts them into format suitable for encoding pass

        Args:
            x (np.ndarray): Raw image(s), either shape (H, W, C) or batched shape (B, H, W, C). Note: This assumes the
                input is in np.uint8 form!

        Returns:
            torch.Tensor: Preprocessed set of images, in shape (B, C, H, W)
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Runs the raw forward pass for this model

        Args:
            x (torch.Tensor): Input tensor in shape (B, C, H, W)

        Returns:
            torch.Tensor: Raw set of features from the model
        """
        raise NotImplementedError

    def get_features(self, x):
        """
        Encodes input image @x into a set a features

        Args:
            x (np.ndarray): Either 3-dimensional (H, W, C) image or 4-dimensional (B, H, W, C) batched set of images to
                encode

        Returns:
            np.ndarray: Generated set of features, where the final dimension is the feature dimension
        """
        # NOTE: ASSUMES RGB INPUT in np.uint8 form
        # ASSUMES SHAPE (B, H, W, C) which is internally mapped to (B, C, H, W)
        assert x.dtype == np.uint8, "Expected input images' dtype to be np.uint8!"
        assert x.shape[-1] == 3, "Expected input images' final dimension to be channels (RGB) = 3!"
        assert len(x.shape) == 3 or len(x.shape) == 4, \
            "Expected inputted image(s) to be 3-dim (H, W, C) or batched 4-dim (B, H, W, C)!"

        # We're not training, so use inference (eval) mode
        with torch.inference_mode():

            # Preprocess images
            if len(x.shape) == 4 and self.preprocess_batch_size is not None:
                # Calculate the number of chunks needed
                n_batch_preprocess_img = int(np.ceil(x.shape[0] / self.preprocess_batch_size))

                # Process each chunk
                preprocessed_chunks = []
                for batch_idx in range(n_batch_preprocess_img):
                    start = batch_idx * self.max_batch_size_preprocess_img
                    end = min((batch_idx + 1) * self.max_batch_size_preprocess_img, x.shape[0])
                    chunk = x[start:end]

                    # Preprocess the current chunk and move it to the device
                    preprocessed_chunk = self.preprocess(chunk).to(self.device)
                    preprocessed_chunks.append(preprocessed_chunk)

                # Concatenate all preprocessed chunks along the batch dimension
                preprocessed_images = torch.cat(preprocessed_chunks, dim=0).to(self.device)
            else:
                preprocessed_images = self.preprocess(x).to(self.device)
            B, C, H, W = preprocessed_images.shape

            # Encode images
            if self.batch_size is not None:
                n_batches = int(np.ceil(B / self.batch_size))
                feature_batches = []
                for i in range(n_batches):
                    start = i * self.batch_size
                    end = min((i + 1) * self.batch_size, B)
                    batch = preprocessed_images[start:end]
                    feature_batch = self.forward(batch).detach().cpu().numpy()
                    feature_batches.append(feature_batch)
                features = np.concatenate(feature_batches, axis=0)
            else:
                features = self.forward(preprocessed_images).detach().cpu().numpy()

        return features

    @property
    def embedding_dim(self):
        """
        Returns:
            int: Outputted embedding dimension for a given input
        """
        raise NotImplementedError



