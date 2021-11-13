import os
import abc
import math
import luigi
import zipfile
import requests
from tqdm import tqdm

OUTPUT_PATH = os.path.join(os.getcwd(), "data/")
DATASETS = {
    "ml-1m": ["https://files.grouplens.org/datasets/movielens/ml-1m.zip"],
    "ml-100k": ["https://files.grouplens.org/datasets/movielens/ml-100k.zip"],
    "ml-25m": ["https://files.grouplens.org/datasets/movielens/ml-25m.zip"],
}


class DownloadDataset(luigi.Task, metaclass=abc.ABCMeta):
    output_path: str = luigi.Parameter(default=OUTPUT_PATH)
    dataset: str = luigi.ChoiceParameter(choices=DATASETS.keys())

    def output(self):
        return [
            luigi.LocalTarget(
                os.path.join(self.output_path, self.dataset, os.path.basename(p))
            )
            for p in DATASETS[self.dataset]
        ]

    def run(self):
        self.load_dataset(self.dataset, output_path=self.output_path)

    def load_dataset(self, name, cache=True, output_path=".", **kws):
        results = []
        for url in DATASETS[name]:

            output_file = os.path.join(output_path, name, os.path.basename(url))
            if not os.path.isfile(output_file) or not cache:
                # Streaming, so we can iterate over the response.
                r = requests.get(url, stream=True)

                # Total size in bytes.
                total_size = int(r.headers.get("content-length", 0))
                block_size = 1024
                wrote = 0
                os.makedirs(os.path.split(output_file)[0], exist_ok=True)
                with open(output_file, "wb") as f:
                    for data in tqdm(
                        r.iter_content(block_size),
                        total=math.ceil(total_size // block_size),
                        unit="KB",
                        unit_scale=True,
                    ):
                        wrote = wrote + len(data)
                        f.write(data)
                if total_size != 0 and wrote != total_size:
                    raise ConnectionError("ERROR, something went wrong")

            if output_file.endswith(".zip"):
                with zipfile.ZipFile(output_file, "r") as zip_ref:
                    zip_ref.extractall(output_path)
