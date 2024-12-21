import torch
from geopy.distance import geodesic

from src.metrics.base_metric import BaseMetric


class ThresholdMetric(BaseMetric):
    def __init__(self, threshold, gps_gallery, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.gps_gallery = gps_gallery

    def __call__(self, logits: torch.Tensor, locations: torch.Tensor, **kwargs):
        gallery_classes = logits.argmax(dim=-1)

        accuracy = 0
        for i in range(len(locations)):
            distance = geodesic(self.gps_gallery[gallery_classes[i]], locations[i]).km
            if distance <= self.threshold:
                accuracy += 1
        accuracy = accuracy / len(locations)

        return accuracy
