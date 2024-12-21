import torch


def equal_earth_projection(locations):
    P1 = 1.340264
    P2 = -0.081106
    P3 = 0.000893
    P4 = 0.003796
    SF = 66.50336

    latitudes, longitudes = locations[:, 0], locations[:, 1]

    latitudes_rad = torch.deg2rad(latitudes)
    longitudes_rad = torch.deg2rad(longitudes)

    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitudes_rad)
    theta = torch.asin(sin_theta)

    numerator = 2 * torch.sqrt(torch.tensor(3.0)) * longitudes_rad * torch.cos(theta)
    denominator = 3 * (
        9 * P4 * theta**8 + 7 * P3 * theta**6 + 3 * P2 * theta**2 + P1
    )
    transformed_latitudes = numerator / denominator

    transformed_longitudes = (
        P4 * theta**9 + P3 * theta**7 + P2 * theta**3 + P1 * theta
    )

    return (
        torch.stack((transformed_latitudes, transformed_longitudes), dim=1) * SF
    ) / 180
