import warnings

import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.image_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.head = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        x = self.clip_model.get_image_features(pixel_values=x)
        x = self.head(x)
        return x
