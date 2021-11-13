import os
import luigi
import yaml
import json

from src.data.datasets import (
    ML1MLoadAndPrepareDataset,
    ML100kLoadAndPrepareDataset,
    ML25MLoadAndPrepareDataset,
)


## Available versions of the GenerateDataset subtasks
DATASETS = dict(
    movie_lens_1m=ML1MLoadAndPrepareDataset,
    movie_lens_100k=ML100kLoadAndPrepareDataset,
    movie_lens_25m=ML25MLoadAndPrepareDataset,
)

OUTPUT_PATH = os.path.join(os.getcwd(), "data/")


class DatasetGeneration(luigi.Task):
    dataset_version: str = luigi.ChoiceParameter(choices=DATASETS.keys())

    def output(self):
        return luigi.LocalTarget(
            os.path.join("data", str(self.dataset_version + "_output_path.json"))
        )

    def run(self):
        dataset = yield DATASETS[self.dataset_version](
            output_path=OUTPUT_PATH, **self.dataset_config
        )

        _output = {}
        for data in dataset:
            _output[data] = dataset[data].path

        with open(self.output().path, "w") as file:
            json.dump(_output, file)

    @property
    def dataset_config(self):
        path = os.path.abspath(
            os.path.join("data", "{}.yaml".format(self.dataset_version))
        )

        with open(path) as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)

        return train_config
