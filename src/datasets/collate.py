import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    batch_size = len(dataset_items)

    result_batch = {}

    result_batch["images"] = torch.stack(
        [elem["image"] for elem in dataset_items], dim=0
    )

    latitudes = torch.tensor([elem["latitude"] for elem in dataset_items])
    longitudes = torch.tensor([elem["longitude"] for elem in dataset_items])
    result_batch["locations"] = torch.stack((latitudes, longitudes), dim=1)

    result_batch["labels"] = torch.tensor([i for i in range(batch_size)]).long()

    return result_batch
