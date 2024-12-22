import torch
import torch.nn as nn

from src.model.location_encoder.equal_earth_projection import equal_earth_projection
from src.model.location_encoder.gaussian_encoding import GaussianEncoding


class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma, rff_vector_size=256, hidden_size=1024):
        super().__init__()

        rff_encoding = GaussianEncoding(
            sigma=sigma, input_size=2, encoded_size=rff_vector_size
        )

        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(2 * rff_vector_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.head = nn.Sequential(nn.Linear(hidden_size, 2 * rff_vector_size))

    def forward(self, x):
        return self.head(self.capsule(x))


class LocationEncoder(nn.Module):
    def __init__(self, sigma_list=[2**0, 2**4, 2**8]):
        super().__init__()

        self.sigma_list = sigma_list
        self.sigma_num = len(self.sigma_list)

        for i, sigma in enumerate(self.sigma_list):
            self.add_module("LocEnc" + str(i), LocationEncoderCapsule(sigma=sigma))

    def forward(self, location):
        eep_location = equal_earth_projection(location)

        location_features = torch.zeros(
            eep_location.shape[0], 512, device=location.device
        )
        for i in range(self.sigma_num):
            location_features += self._modules["LocEnc" + str(i)](eep_location)

        return location_features
