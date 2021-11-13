import os
from tqdm import tqdm

import torch
import numpy as np

from src.model.pmf import PMF
from src.model.actor import Actor
from src.model.critic import Critic
from src.model.ou_noise import OUNoise
from src.model.replay_buffer import PriorityExperienceReplay
from src.model.state_representation import (
    DRRAveStateRepresentation,
    FairRecStateRepresentation,
)

import wandb


STATE_REPRESENTATION = dict(
    movie_lens_100k=DRRAveStateRepresentation,
    movie_lens_100k_fair=FairRecStateRepresentation,
)


class DRRAgent:
    def __init__(
        self,
        env,
        users_num,
        items_num,
        state_size,
        srm_size,
        model_path,
        embedding_network_weights_path,
        emb_model,
        train_version,
        is_test=False,
        use_wandb=False,
        embedding_dim=100,
        actor_hidden_dim=128,
        actor_learning_rate=0.001,
        critic_hidden_dim=128,
        critic_learning_rate=0.001,
        discount_factor=0.9,
        tau=0.001,
        learning_starts=1000,
        replay_memory_size=1000000,
        batch_size=32,
        n_groups=4,
        fairness_constraints=[0.25, 0.25, 0.25, 0.25],
        no_cuda=False,
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        )

        self.env = env

        self.users_num = users_num
        self.items_num = items_num

        self.model_path = model_path

        self.emb_model = emb_model
        self.embedding_network_weights_path = embedding_network_weights_path

        self.embedding_dim = embedding_dim
        self.srm_size = srm_size
        self.actor_hidden_dim = actor_hidden_dim
        self.actor_learning_rate = actor_learning_rate
        self.critic_hidden_dim = critic_hidden_dim
        self.critic_learning_rate = critic_learning_rate
        self.discount_factor = discount_factor
        self.tau = tau

        self.learning_starts = learning_starts
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size

        self.n_groups = n_groups
        self.fairness_constraints = fairness_constraints

        self.actor = Actor(
            self.embedding_dim,
            self.srm_size,
            self.actor_hidden_dim,
            self.actor_learning_rate,
            state_size,
            self.tau,
            self.device,
        )
        self.critic = Critic(
            self.critic_hidden_dim,
            self.critic_learning_rate,
            self.embedding_dim,
            self.srm_size,
            self.tau,
            self.device,
        )

        if self.emb_model == "user_movie":
            self.reward_model = PMF(users_num, items_num, self.embedding_dim).to(
                self.device
            )
            self.reward_model.load_state_dict(
                torch.load(
                    self.embedding_network_weights_path,
                    map_location=torch.device(self.device),
                )
            )

            self.user_embeddings = self.reward_model.user_embeddings.weight.data
            self.item_embeddings = self.reward_model.item_embeddings.weight.data
        else:
            raise "Embedding Model Type not supported"

        if self.env:
            self.env.reward_model = self.reward_model
            self.env.device = self.device

        self.srm_ave = STATE_REPRESENTATION[train_version](
            self.embedding_dim, self.n_groups
        ).to(self.device)

        self.buffer = PriorityExperienceReplay(
            self.replay_memory_size,
            self.embedding_dim,
            self.srm_size * self.embedding_dim,
        )
        self.epsilon_for_priority = 1e-6

        # noise
        self.noise = OUNoise(self.embedding_dim, decay_period=10)

        self.is_test = is_test

        # wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=train_version,
                config={
                    "users_num": users_num,
                    "items_num": items_num,
                    "state_size": state_size,
                    "emb_model": emb_model,
                    "embedding_dim": self.embedding_dim,
                    "actor_hidden_dim": self.actor_hidden_dim,
                    "actor_learning_rate": self.actor_learning_rate,
                    "critic_hidden_dim": self.critic_hidden_dim,
                    "critic_learning_rate": self.critic_learning_rate,
                    "discount_factor": self.discount_factor,
                    "tau": self.tau,
                    "learning_starts": self.learning_starts,
                    "replay_memory_size": self.replay_memory_size,
                    "batch_size": self.batch_size,
                    "group_fairness": n_groups,
                    "fairness_constraints": self.fairness_constraints,
                },
            )

    def calculate_td_target(self, rewards, q_values, dones):
        return rewards + ((1 - dones) * (self.discount_factor * q_values))

    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        if items_ids == None:
            items_ids = list(set(i for i in range(self.items_num)) - recommended_items)

        items_ids = np.array(items_ids)
        items_ebs = self.get_items_emb(items_ids)
        action = torch.transpose(action, 1, 0).float()

        if top_k:
            item_indice = torch.argsort(
                torch.transpose(torch.matmul(items_ebs, action), 1, 0)
            )[0][-top_k:]
            return items_ids[item_indice.detach().cpu().numpy()]
        else:
            item_idx = torch.argmax(torch.matmul(items_ebs, action))
            return items_ids[item_idx]

    def get_items_emb(self, items_ids):
        if self.emb_model == "user_movie":
            items_eb = self.item_embeddings[items_ids]
        else:
            raise "Emb Model Type not supported"

        return items_eb

    def get_state(self, user_id, items_ids):
        user_eb = self.user_embeddings[user_id]
        items_eb = self.get_items_emb(items_ids)

        with torch.no_grad():
            ## SRM state
            state = self.srm_ave(
                [
                    user_eb.unsqueeze(0),
                    items_eb.unsqueeze(0),
                ]
            )

        return state

    def train(self, max_episode_num, top_k=False, load_model=False):
        self.actor.update_target_network()
        self.critic.update_target_network()

        if load_model:
            # Get list of checkpoints
            actor_checkpoints = sorted(
                [
                    int(f.split("_")[1])
                    for f in os.listdir(self.model_path)
                    if f.startswith("actor")
                ]
            )
            critic_checkpoints = sorted(
                [
                    int(f.split("_")[1])
                    for f in os.listdir(self.model_path)
                    if f.startswith("critic")
                ]
            )
            self.load_model(
                os.path.join(
                    self.model_path, "actor_{}.h5".format(actor_checkpoints[-1])
                ),
                os.path.join(
                    self.model_path, "critic_{}.h5".format(critic_checkpoints[-1])
                ),
            )
            print("----- Completely load weights!")

        sum_precision = 0
        sum_ndcg = 0
        sum_propfair = 0
        sum_reward = 0

        for episode in tqdm(range(max_episode_num)):

            # episodic reward
            episode_reward = 0
            steps = 0
            critic_loss = 0
            actor_loss = 0
            mean_action = 0
            mean_precision = 0
            mean_ndcg = 0

            # environment
            user_id, items_ids, done = self.env.reset()

            while not done:
                # observe current state & Find action
                state = self.get_state(user_id, items_ids)

                with torch.no_grad():
                    ## action(ranking score)
                    action = self.actor.network(state)

                    ## epsilon-greedy exploration
                    if not self.is_test:
                        action = self.noise.get_action(
                            action.detach().cpu().numpy()[0], steps
                        ).to(self.device)

                ## item
                recommended_item = self.recommend_item(
                    action, self.env.recommended_items, top_k=top_k
                )

                # calculate reward and observe new state
                ## Step
                next_items_ids, reward, done, _ = self.env.step(
                    recommended_item, top_k=top_k
                )

                # get next_state
                next_state = self.get_state(user_id, next_items_ids)

                # buffer
                self.buffer.append(
                    state.detach().cpu().numpy(),
                    action.detach().cpu().numpy(),
                    np.sum(reward) if top_k else reward,
                    next_state.detach().cpu().numpy(),
                    done,
                )

                if self.buffer.crt_idx > self.learning_starts or self.buffer.is_full:
                    _critic_loss, _actor_loss = self.update_model()
                    actor_loss += _actor_loss
                    critic_loss += _critic_loss

                items_ids = next_items_ids
                episode_reward += np.sum(reward) if top_k else reward

                mean_action += np.sum(action[0].cpu().numpy()) / (
                    len(action[0].cpu().numpy())
                )
                steps += 1

                if top_k:
                    correct_list = [1 if r > 0 else 0 for r in reward]
                    # ndcg
                    dcg, idcg = self.calculate_ndcg(
                        correct_list, [1 for _ in range(len(reward))]
                    )
                    mean_ndcg += dcg / idcg

                    # precision
                    correct_num = top_k - correct_list.count(0)
                    mean_precision += correct_num / top_k
                else:
                    mean_precision += 1 if reward > 0 else 0

                if done:
                    propfair = 0
                    total_exp = np.sum(list(self.env.group_count.values()))
                    if total_exp > 0:
                        propfair = np.sum(
                            np.array(self.fairness_constraints)
                            * np.log(
                                1
                                + np.array(list(self.env.group_count.values()))
                                / total_exp
                            )
                        )

                    sum_precision += mean_precision / steps
                    sum_ndcg += mean_ndcg / steps
                    sum_propfair += propfair
                    sum_reward += episode_reward

                    if self.use_wandb:
                        wandb.log(
                            {
                                "precision": (mean_precision / steps) * 100,
                                "ndcg": mean_ndcg / steps,
                                "total_reward": episode_reward,
                                "critic_loss": critic_loss / steps,
                                "actor_loss": actor_loss / steps,
                                "mean_action": mean_action / steps,
                                "propfair": propfair,
                                "cvr": mean_precision / steps,
                                "ufg": propfair
                                / max(1 - (mean_precision / steps), 0.01),
                            }
                        )

            if (episode + 1) % 1000 == 0:
                self.save_model(
                    os.path.join(self.model_path, "actor_{}.h5".format(episode + 1)),
                    os.path.join(self.model_path, "critic_{}.h5".format(episode + 1)),
                )

        return (
            sum_precision / max_episode_num,
            sum_ndcg / max_episode_num,
            sum_propfair / max_episode_num,
            sum_reward / max_episode_num,
        )

    def update_model(self):
        # sample a minibatch
        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
            weight_batch,
            index_batch,
        ) = self.buffer.sample(self.batch_size)

        batch_states = torch.FloatTensor(batch_states).to(self.device)
        batch_actions = torch.FloatTensor(batch_actions).to(self.device)
        batch_rewards = torch.FloatTensor(batch_rewards).to(self.device)
        batch_next_states = torch.FloatTensor(batch_next_states).to(self.device)
        batch_dones = torch.FloatTensor(batch_dones).to(self.device)
        weight_batch = torch.FloatTensor(weight_batch).to(self.device)

        # set TD targets
        target_next_action = self.actor.target_network(batch_next_states)

        qs = self.critic.network(
            [
                target_next_action,
                batch_next_states,
            ]
        )

        target_qs = self.critic.target_network(
            [
                target_next_action,
                batch_next_states,
            ]
        )

        min_qs = torch.min(torch.cat([target_qs, qs], axis=1), 1, True).values.squeeze(
            1
        )  # Double Q method

        td_targets = self.calculate_td_target(
            batch_rewards,
            min_qs,
            batch_dones,
        ).unsqueeze(1)

        # update priority
        for (p, i) in zip(td_targets, index_batch):
            self.buffer.update_priority(abs(p[0]) + self.epsilon_for_priority, i)

        # update critic network
        value = self.critic.network(
            [
                batch_actions,
                batch_states,
            ]
        )
        value_loss = self.critic.loss(value, td_targets)
        weighted_loss = torch.mean(value_loss * weight_batch)

        self.critic.optimizer.zero_grad()
        weighted_loss.backward(retain_graph=True)
        self.critic.optimizer.step()

        # update actor network
        loss = self.critic.network(
            [
                batch_actions,
                batch_states,
            ]
        )
        loss = -loss.mean()

        self.actor.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # target update
        self.actor.update_target_network()
        self.critic.update_target_network()

        return weighted_loss.detach().cpu().numpy(), loss.detach().cpu().numpy()

    def evaluate(self, env, top_k=0, available_items=None):
        # episodic reward
        episode_reward = 0
        steps = 0

        mean_precision = 0
        mean_ndcg = 0

        # Environment
        user_id, items_ids, done = env.reset()

        while not done:
            # Observe current state and Find action
            state = self.get_state(user_id, items_ids)

            with torch.no_grad():
                ## Action(ranking score)
                action = self.actor.network(state)

            ## Item
            recommended_item = self.recommend_item(
                action,
                env.recommended_items,
                top_k=top_k,
                items_ids=list(available_items),
            )

            # Calculate reward and observe new state (in env)
            ## Step
            next_items_ids, reward, done, _ = env.step(recommended_item, top_k=top_k)

            if top_k:
                correct_list = [1 if r > 0 else 0 for r in reward]
                # ndcg
                dcg, idcg = self.calculate_ndcg(
                    correct_list, [1 for _ in range(len(reward))]
                )
                mean_ndcg += dcg / idcg

                # precision
                correct_num = top_k - correct_list.count(0)
                mean_precision += correct_num / top_k
            else:
                mean_precision += 1 if reward > 0 else 0

            reward = np.sum(reward)
            items_ids = next_items_ids
            episode_reward += reward
            steps += 1
            available_items = (
                available_items - set(recommended_item) if available_items else None
            )

        mean_precision = mean_precision / steps
        mean_ndcg = mean_ndcg / steps

        propfair = 0
        total_exp = np.sum(list(env.group_count.values()))
        if total_exp > 0:
            propfair = np.sum(
                np.array(self.fairness_constraints)
                * np.log(1 + np.array(list(env.group_count.values())) / total_exp)
            )

        return (mean_precision, mean_ndcg, propfair)

    def calculate_ndcg(self, rel, irel):
        dcg = 0
        idcg = 0
        rel = [1 if r > 0 else 0 for r in rel]
        for i, (r, ir) in enumerate(zip(rel, irel)):
            dcg += (r) / np.log2(i + 2)
            idcg += (ir) / np.log2(i + 2)

        return dcg, idcg

    def save_model(self, actor_path, critic_path, buffer_path=None):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        if buffer_path:
            import pickle

            with open(buffer_path, "wb") as f:
                pickle.dump(self.buffer, f)

    def load_model(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
