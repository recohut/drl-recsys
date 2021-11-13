import os
from typing import Optional, Tuple, Union

from tqdm import tqdm
from pathlib import Path
import requests
import zipfile
import math
import numpy as np
import pandas as pd
import torch

from sklearn import preprocessing
from sklearn.utils import check_random_state

from obp.dataset import BaseRealBanditDataset
from obp.types import BanditFeedback

from src.model.pmf import PMF

DATASETS = {
    "ml-100k": ["https://files.grouplens.org/datasets/movielens/ml-100k.zip"],
}


class MovieLensDataset(BaseRealBanditDataset):
    def __init__(
        self,
        data_path: str,
        embedding_network_weights_path: str,
        embedding_dim: int,
        users_num: int,
        items_num: int,
        state_size: int,
        filter_ids: list = None,
    ):
        """Dataset container"""

        reward_model = PMF(users_num, items_num, embedding_dim)
        reward_model.load_state_dict(
            torch.load(embedding_network_weights_path, map_location=torch.device("cpu"))
        )
        self.user_embeddings = reward_model.user_embeddings.weight.data
        self.item_embeddings = reward_model.item_embeddings.weight.data

        self.state_size = state_size

        self.data_path = os.path.join(data_path, "ml-100k")
        # self.download_data("ml-100k", data_path)
        self.load_raw_data(filter_ids)
        self.pre_process()

    @property
    def n_rounds(self) -> int:
        """Total number of rounds contained in the logged bandit dataset."""
        return self.data.shape[0]

    @property
    def n_actions(self) -> int:
        """Number of actions."""
        return int(self.action.max() + 1)

    @property
    def dim_context(self) -> int:
        """Dimensions of context vectors."""
        return self.context.shape[1]

    @property
    def len_list(self) -> int:
        """Length of recommendation lists."""
        return 1

    def download_data(self, name, output_path, cache=True) -> None:
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

            print(output_file)
            if output_file.endswith(".zip"):
                with zipfile.ZipFile(output_file, "r") as zip_ref:
                    zip_ref.extractall(output_path)

    def load_raw_data(self, filter_ids=None) -> None:
        """Load raw open bandit dataset."""

        ratings_df = pd.read_csv(
            os.path.join(self.data_path, "u.data"),
            "\t",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
            dtype=np.uint32,
        )
        users_df = pd.read_csv(
            os.path.join(self.data_path, "u.user"),
            "|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
            engine="python",
        )
        movies_list = [
            i.strip().split("|")
            for i in open(
                os.path.join(self.data_path, "u.item"), encoding="latin-1"
            ).readlines()
        ]
        movies_df = pd.DataFrame(
            movies_list,
            columns=[
                "movie_id",
                "title",
                "release_date",
                "video_release_date",
                "imdb_url",
                "unknown",
                "Action",
                "Adventure",
                "Animation",
                "Childrens",
                "Comedy",
                "Crime",
                "Documentary",
                "Drama",
                "Fantasy",
                "Film_Noir",
                "Horror",
                "Musical",
                "Mystery",
                "Romance",
                "Sci-Fi",
                "Thriller",
                "War",
                "Western",
            ],
        )
        movies_df["movie_id"] = movies_df["movie_id"].apply(pd.to_numeric)

        movies_encoder = preprocessing.LabelEncoder()
        movies_encoder.fit(movies_df["movie_id"].values)
        movies_df["movie_id"] = movies_encoder.transform(movies_df["movie_id"].values)
        ratings_df["movie_id"] = movies_encoder.transform(ratings_df["movie_id"].values)

        users_encoder = preprocessing.LabelEncoder()
        users_encoder.fit(users_df["user_id"].values)
        users_df["user_id"] = users_encoder.transform(users_df["user_id"].values)
        ratings_df["user_id"] = users_encoder.transform(ratings_df["user_id"].values)

        if filter_ids:
            ratings_df = ratings_df[ratings_df["user_id"].isin(filter_ids)]

        ratings_df = ratings_df.applymap(int)
        ratings_df = ratings_df.sort_values("timestamp")

        movies_df = movies_df.drop(
            columns=["release_date", "video_release_date", "imdb_url"]
        )

        # Train Dataset
        self.data = ratings_df
        self.item_context = movies_df

        print("----- Finished data load")

    def pre_process(self) -> None:
        """Preprocess raw open bandit dataset."""

        # Train Dataset
        print("----- Preprocessing dataset")

        _df = self.data[self.data["rating"] >= 4].copy()
        df_list = []
        for g, df_g in _df.groupby("user_id"):
            list_of_indexes = []
            df_g["movie_id"].rolling(self.state_size, min_periods=1).apply(
                (lambda x: list_of_indexes.append(x.astype(int).tolist()) or 0),
                raw=False,
            )
            df_g["item_id_history"] = list_of_indexes
            df_g["item_id_history"] = df_g["item_id_history"].shift(1)

            Item_prev = df_g["item_id_history"].apply(
                lambda x: [] if x is np.nan else x
            )
            df_list.append(Item_prev)

        _df["item_id_history"] = pd.concat(df_list)
        self.data["item_id_history"] = _df["item_id_history"]

        self.data["item_id_history"] = self.data.groupby(["user_id"])[
            "item_id_history"
        ].apply(lambda x: x.bfill().ffill())

        self.data["item_id_history"] = self.data["item_id_history"].apply(
            lambda x: x if type(x) is list else []
        )

        self.data["item_id_history"] = self.data["item_id_history"].apply(
            lambda x: np.array(x)
        )

        self.data["len_history"] = self.data["item_id_history"].apply(
            lambda x: x.shape[0]
        )
        self.data = self.data.drop(self.data[self.data["len_history"] < 5].index)

        item_ave = np.array(
            [
                np.mean(self.item_embeddings[item].cpu().numpy(), 0)
                for item in self.data["item_id_history"].values
            ]
        )

        self.context = np.concatenate(
            (
                self.user_embeddings[self.data["user_id"].values].cpu().numpy(),
                self.user_embeddings[self.data["user_id"].values].cpu().numpy()
                * item_ave,
                item_ave,
            ),
            axis=1,
        )

        self.action_context = (
            self.item_embeddings[self.item_context["movie_id"].values].cpu().numpy()
        )

        self.action = self.data["movie_id"].values
        self.reward = (
            self.data["rating"].apply(lambda x: 0 if x < 4 else 1).values
        )  # 0.5 * (self.data["rating"].values - 3)
        self.pscore = np.ones_like(self.action, dtype=int) * (
            1 / (self.item_context["movie_id"].max() + 1)
        )

        self.position = np.zeros_like(self.action, dtype=int)

        print("Finished preprocessing")

    def obtain_batch_bandit_feedback(
        self,
        test_size: float = 0.3,
        split: bool = False,
    ) -> Union[BanditFeedback, Tuple[BanditFeedback, BanditFeedback]]:
        """Obtain batch logged bandit feedback.

        Returns
        --------
        bandit_feedback: BanditFeedback
            A dictionary containing batch logged bandit feedback data collected by a behavior policy.
            The keys of the dictionary are as follows.
            - n_rounds: number of rounds (size) of the logged bandit data
            - n_actions: number of actions (:math:`|\mathcal{A}|`)
            - action: action variables sampled by a behavior policy
            - position: positions where actions are recommended
            - reward: reward variables
            - pscore: action choice probabilities by a behavior policy
            - context: context vectors such as user-related features and user-item affinity scores
            - action_context: item-related context vectors

        """
        if split:
            n_rounds_train = np.int(self.n_rounds * (1.0 - test_size))
            bandit_feedback_train = dict(
                n_rounds=n_rounds_train,
                n_actions=self.n_actions,
                action=self.action[:n_rounds_train],
                position=self.position[:n_rounds_train],
                reward=self.reward[:n_rounds_train],
                pscore=self.pscore[:n_rounds_train],
                context=self.context[:n_rounds_train],
                action_context=self.action_context[:n_rounds_train],
            )

            bandit_feedback_test = dict(
                n_rounds=(self.n_rounds - n_rounds_train),
                n_actions=self.n_actions,
                action=self.action[n_rounds_train:],
                position=self.position[n_rounds_train:],
                reward=self.reward[n_rounds_train:],
                pscore=self.pscore[n_rounds_train:],
                context=self.context[n_rounds_train:],
                action_context=self.action_context[n_rounds_train:],
            )

            return bandit_feedback_train, bandit_feedback_test

        else:
            return dict(
                n_rounds=self.n_rounds,
                n_actions=self.n_actions,
                action=self.action,
                position=self.position,
                reward=self.reward,
                pscore=self.pscore,
                context=self.context,
                action_context=self.action_context,
            )

    def sample_bootstrap_bandit_feedback(
        self,
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
        random_state: Optional[int] = None,
    ) -> BanditFeedback:
        """Obtain bootstrap logged bandit feedback.

        Parameters
        -----------
        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        --------
        bandit_feedback: BanditFeedback
            A dictionary containing logged bandit feedback data sampled independently from the original data with replacement.
            The keys of the dictionary are as follows.
            - n_rounds: number of rounds (size) of the logged bandit data
            - n_actions: number of actions
            - action: action variables sampled by a behavior policy
            - position: positions where actions are recommended
            - reward: reward variables
            - pscore: action choice probabilities by a behavior policy
            - context: context vectors such as user-related features and user-item affinity scores
            - action_context: item-related context vectors

        """

        if is_timeseries_split:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )[0]
        else:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )

        n_rounds = bandit_feedback["n_rounds"]
        random_ = check_random_state(random_state)
        bootstrap_idx = random_.choice(np.arange(n_rounds), size=n_rounds, replace=True)
        for key_ in ["action", "position", "reward", "pscore", "context"]:
            bandit_feedback[key_] = bandit_feedback[key_][bootstrap_idx]
        return bandit_feedback

    @classmethod
    def calc_on_policy_policy_value_estimate(
        cls,
        data_path: Optional[Path] = None,
    ) -> float:
        """Calculate on-policy policy value estimate (used as a ground-truth policy value).

        Parameters
        ----------
        data_path: Path, default=None
            Path where the Open Bandit Dataset exists.
            When `None` is given, this class downloads the example small-sized version of the dataset.

        Returns
        ---------
        on_policy_policy_value_estimate: float
            Policy value of the behavior policy estimated by on-policy estimation, i.e., :math:`\\mathbb{E}_{\\mathcal{D}} [r_t]`.
            where :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
            This parameter is used as a ground-truth policy value in the evaluation of OPE estimators.

        """
        bandit_feedback = cls(data_path=data_path).obtain_batch_bandit_feedback()
        bandit_feedback_test = bandit_feedback[1]
        return bandit_feedback_test["reward"].mean()
