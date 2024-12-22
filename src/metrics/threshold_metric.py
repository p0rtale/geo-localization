import torch
from geopy.distance import geodesic

from src.metrics.base_metric import BaseMetric


class ThresholdMetric(BaseMetric):
    def __init__(self, threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def __call__(self, pred_locations: torch.Tensor, locations: torch.Tensor, **kwargs):
        accuracy = 0
        for i in range(len(locations)):
            distance = geodesic(pred_locations[i], locations[i]).km
            if distance <= self.threshold:
                accuracy += 1
        accuracy = accuracy / len(locations)

        return accuracy
