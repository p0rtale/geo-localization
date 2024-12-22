import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.model.image_encoder import ImageEncoder
from src.model.location_encoder import LocationEncoder
from src.utils.io_utils import ROOT_PATH


class GeoCLIP(nn.Module):
    def __init__(
        self,
        train_queue_size=4096,
        val_queue_size=4096,
        gps_gallery_filename="coordinates_100K.csv",
    ):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()

        self.gps_gallery = self._load_gps_data(
            ROOT_PATH / "src" / "model" / "gps_gallery" / gps_gallery_filename
        )

        self.train_queue_size = train_queue_size
        self.register_buffer("train_queue", torch.randn(train_queue_size, 2))
        self.train_queue = nn.functional.normalize(self.train_queue, dim=1)
        self.register_buffer("train_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.val_queue_size = val_queue_size
        self.register_buffer("val_queue", torch.randn(val_queue_size, 2))
        self.val_queue = nn.functional.normalize(self.val_queue, dim=1)
        self.register_buffer("val_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.device = "cpu"

    def to(self, device):
        self.device = device

        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        self.gps_gallery = self.gps_gallery.to(device)

        return super().to(device)

    def _load_gps_data(self, path):
        data = pd.read_csv(path)
        lat_lon = data[["LAT", "LON"]]
        return torch.tensor(lat_lon.values, dtype=torch.float32)

    @torch.no_grad()
    def update_queue(self, gps_batch):
        """
        Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        if self.training:
            gps_queue = self.train_queue
            queue_size = self.train_queue_size
            gps_queue_ptr = self.train_queue_ptr
        else:
            gps_queue = self.val_queue
            queue_size = self.val_queue_size
            gps_queue_ptr = self.val_queue_ptr

        gps_batch_size = gps_batch.shape[0]
        gps_ptr = int(gps_queue_ptr)

        # TODO: deal with the latest validation batch
        # assert queue_size % gps_batch_size == 0, \
        #        f"Queue size {queue_size} should be divisible by batch size {gps_batch_size}"

        gps_queue[gps_ptr : gps_ptr + gps_batch_size, :] = gps_batch
        gps_ptr = (gps_ptr + gps_batch_size) % queue_size
        gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        if self.training:
            return self.train_queue
        return self.val_queue

    def get_gps_gallery(self):
        return self.gps_gallery

    def forward(self, images, locations, **batch):
        """
        GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)

        Returns:
            logits_per_image (torch.Tensor): Logits per image of shape (n, m)
        """

        # Compute Features
        image_features = self.image_encoder(images)
        location_features = self.location_encoder(locations)
        logit_scale = self.logit_scale.exp()

        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)

        # Cosine similarity (Image Features & Location Features)
        logits_per_image = logit_scale * (image_features @ location_features.t())

        return {"logits": logits_per_image}

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """
        Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return

        Returns:
            top_pred_gps (torch.Tensor): Top k GPS coordinates of shape (k, 2)
            top_pred_prob (torch.Tensor): Top k GPS probabilities of shape (k,)
        """
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        logits_per_image = self.forward(image, self.gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top k predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
