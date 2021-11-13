import os
import json
import pickle
import luigi
import wandb

from src.environment.ml_env import OfflineEnv, OfflineFairEnv
from src.model.recommender import DRRAgent, FairRecAgent

AGENT = dict(drr=DRRAgent, fairrec=FairRecAgent)
ENV = dict(drr=OfflineEnv, fairrec=OfflineFairEnv)


class MovieLens(luigi.Task):
    output_path = luigi.Parameter()

    algorithm: str = luigi.ChoiceParameter(choices=AGENT.keys())
    epochs: int = luigi.IntParameter(default=5)
    users_num: int = luigi.IntParameter(default=6041)
    items_num: int = luigi.IntParameter(default=3953)
    state_size: int = luigi.IntParameter(default=10)
    srm_size: int = luigi.IntParameter(default=3)
    max_eps_num: int = luigi.IntParameter(default=8000)
    embedding_dim: int = luigi.IntParameter(default=100)
    actor_hidden_dim: int = luigi.IntParameter(default=128)
    actor_learning_rate: float = luigi.FloatParameter(default=0.001)
    critic_hidden_dim: int = luigi.IntParameter(default=128)
    critic_learning_rate: float = luigi.FloatParameter(default=0.001)
    discount_factor: float = luigi.FloatParameter(default=0.9)
    tau: float = luigi.FloatParameter(default=0.001)
    learning_starts: int = luigi.IntParameter(default=1000)
    replay_memory_size: int = luigi.IntParameter(default=1000000)
    batch_size: int = luigi.IntParameter(default=32)
    n_groups: int = luigi.IntParameter(default=4)
    fairness_constraints: list = luigi.ListParameter(default=[0.25, 0.25, 0.25, 0.25])
    top_k: int = luigi.IntParameter(default=10)
    done_count: int = luigi.IntParameter(default=10)

    emb_model: str = luigi.Parameter(default="user_movie")
    embedding_network_weights: str = luigi.Parameter(default="")

    train_version: str = luigi.Parameter()
    use_wandb: bool = luigi.BoolParameter()
    load_model: bool = luigi.BoolParameter()
    dataset_path: str = luigi.Parameter()
    evaluate: bool = luigi.BoolParameter()

    def output(self):
        return {
            "actor_model": luigi.LocalTarget(
                os.path.join(self.output_path, "actor.h5")
            ),
            "critic_model": luigi.LocalTarget(
                os.path.join(self.output_path, "critic.h5")
            ),
        }

    def run(self):
        dataset = self.load_data()
        self.train_model(dataset)

    def load_data(self):
        with open(self.dataset_path) as json_file:
            _dataset_path = json.load(json_file)

        dataset = {}
        with open(_dataset_path["train_users_dict"], "rb") as pkl_file:
            dataset["train_users_dict"] = pickle.load(pkl_file)

        with open(_dataset_path["train_users_history_lens"], "rb") as pkl_file:
            dataset["train_users_history_lens"] = pickle.load(pkl_file)

        with open(_dataset_path["eval_users_dict"], "rb") as pkl_file:
            dataset["eval_users_dict"] = pickle.load(pkl_file)

        with open(_dataset_path["eval_users_history_lens"], "rb") as pkl_file:
            dataset["eval_users_history_lens"] = pickle.load(pkl_file)

        with open(_dataset_path["users_history_lens"], "rb") as pkl_file:
            dataset["users_history_lens"] = pickle.load(pkl_file)

        with open(_dataset_path["movies_groups"], "rb") as pkl_file:
            dataset["movies_groups"] = pickle.load(pkl_file)

        return dataset

    def train_model(self, dataset):
        print("---------- Prepare Env")
        print("------------ ", self.algorithm)

        env = ENV[self.algorithm](
            users_dict=dataset["train_users_dict"],
            users_history_lens=dataset["train_users_history_lens"],
            n_groups=self.n_groups,
            movies_groups=dataset["movies_groups"],
            state_size=self.state_size,
            done_count=self.done_count,
            fairness_constraints=self.fairness_constraints,
        )

        print("---------- Initialize Agent")
        recommender = AGENT[self.algorithm](
            env=env,
            users_num=self.users_num,
            items_num=self.items_num,
            srm_size=self.srm_size,
            state_size=self.state_size,
            train_version=self.train_version,
            use_wandb=self.use_wandb,
            embedding_dim=self.embedding_dim,
            actor_hidden_dim=self.actor_hidden_dim,
            actor_learning_rate=self.actor_learning_rate,
            critic_hidden_dim=self.critic_hidden_dim,
            critic_learning_rate=self.critic_learning_rate,
            discount_factor=self.discount_factor,
            tau=self.tau,
            learning_starts=self.learning_starts,
            replay_memory_size=self.replay_memory_size,
            batch_size=self.batch_size,
            model_path=self.output_path,
            emb_model=self.emb_model,
            embedding_network_weights_path=self.embedding_network_weights,
            n_groups=self.n_groups,
            fairness_constraints=self.fairness_constraints,
        )

        print("---------- Start Training")
        recommender.train(
            max_episode_num=self.max_eps_num, top_k=False, load_model=self.load_model
        )

        print("---------- Finish Training")
        print("---------- Saving Model")
        recommender.save_model(
            self.output()["actor_model"].path,
            self.output()["critic_model"].path,
            os.path.join(self.output_path, "buffer.pkl"),
        )
        print("---------- Finish Saving Model")
