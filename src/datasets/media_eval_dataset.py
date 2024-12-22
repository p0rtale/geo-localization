import json
import logging
import os
import random
import shutil
from pathlib import Path

import pandas
import requests

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)

URL_LINKS = {
    "mp16-images": "http://www.cis.jhu.edu/~shraman/TransLocator/Datasets/mp16_rgb_images.tgz",
    "mp16-annotations": "http://www.cis.jhu.edu/~shraman/TransLocator/Annotations.zip",
}


class MediaEvalDataset(BaseDataset):
    def __init__(self, part, split=0.99, data_dir=None, *args, **kwargs):
        assert part in ("train", "val")
        self.part = part
        self.split = split

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "mp16"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = ROOT_PATH / data_dir
        self._images_dir = (
            self._data_dir / "r13_local_data" / "mp16" / "mp16_rgb_images"
        )
        self._annotations_path = self._data_dir / "annotations.csv"

        logger.info("Building MP16 index...")
        index = self._get_or_load_index()
        logger.info("MP16 dataset is ready")

        super().__init__(index, *args, **kwargs)

    def _load_annotations(self):
        arch_path = self._data_dir / "annotations.zip"
        self._load_data(URL_LINKS["mp16-annotations"], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        shutil.move(
            str(self._data_dir / "Annotations" / "mp16_places365.csv"),
            str(self._annotations_path),
        )
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "Annotations"))

    def _load_images(self):
        arch_path = self._data_dir / "images.zip"
        self._load_data(URL_LINKS["mp16-images"], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        os.remove(str(arch_path))

    def _load_data(self, link, path):
        response = requests.get(link, stream=True)
        response.raise_for_status()

        with open(path, "wb") as f:
            f.write(response.content)

    def _create_index(self):
        if not self._annotations_path.exists():
            self._load_annotations()

        annotations_df = pandas.read_csv(
            self._annotations_path,
            usecols=["IMG_ID", "LAT", "LON"],
            dtype={"LAT": "float64", "LON": "float64"},
        )
        annotations_df.rename(
            columns={"IMG_ID": "path", "LAT": "latitude", "LON": "longitude"},
            inplace=True,
        )
        annotations_df["path"] = annotations_df["path"].apply(
            lambda path: f"{str(self._images_dir.absolute().resolve())}/{path}"
        )

        annotations_df = annotations_df[
            annotations_df.apply(lambda x: Path(x["path"]).exists(), axis=1)
        ]

        annotations_list = annotations_df.to_dict(orient="records")

        random.seed(42)
        random.shuffle(annotations_list)

        split_size = int(self.split * len(annotations_list))
        if self.part == "train":
            return annotations_list[:split_size]

        return annotations_list[split_size:]

    def _get_or_load_index(self):
        if not self._images_dir.exists():
            self._load_images()

        index_path = self._data_dir / f"{self.part}_index.json"
        if index_path.exists():
            with index_path.open() as file:
                index = json.load(file)
        else:
            index = self._create_index()
            with index_path.open("w") as file:
                json.dump(index, file, indent=2)

        return index
