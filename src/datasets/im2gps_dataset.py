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
    "im2gps3k-images": "http://www.cis.jhu.edu/~shraman/TransLocator/Datasets/im2gps3k_rgb_images.tar.gz",
    "im2gps3k-annotations": "https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/meta/im2gps3k_places365.csv",
}


class Im2GPSDataset(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "im2gps3k"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = ROOT_PATH / data_dir
        self._images_dir = self._data_dir / "im2gps3k_rgb_images"
        self._annotations_path = self._data_dir / "annotations.csv"

        logger.info("Building Im2GPS3k index...")
        index = self._get_or_load_index()
        logger.info("Im2GPS3k dataset is ready")

        super().__init__(index, *args, **kwargs)

    def _load_annotations(self):
        self._load_data(URL_LINKS["im2gps3k-annotations"], str(self._annotations_path))

    def _load_images(self):
        arch_path = self._data_dir / "images.tar.gz"
        self._load_data(URL_LINKS["im2gps3k-images"], str(arch_path))
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
        return annotations_list

    def _get_or_load_index(self):
        if not self._images_dir.exists():
            self._load_images()

        index_path = self._data_dir / "index.json"
        if index_path.exists():
            with index_path.open() as file:
                index = json.load(file)
        else:
            index = self._create_index()
            with index_path.open("w") as file:
                json.dump(index, file, indent=2)

        return index
