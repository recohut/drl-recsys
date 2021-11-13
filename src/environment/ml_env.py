import torch
import numpy as np


class OfflineEnv(object):
    def __init__(
        self,
        users_dict,
        users_history_lens,
        n_groups,
        movies_groups,
        state_size,
        done_count,
        fairness_constraints,
        fix_user_id=None,
        reward_model=None,
        use_only_reward_model=False,
        device="cpu",
    ):

        self.device = device

        # users: interacted items, rate
        self.users_dict = users_dict
        self.users_history_lens = users_history_lens

        self.state_size = state_size

        # filter users with len_history > state_size
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id

        self.user = (
            fix_user_id if fix_user_id else np.random.choice(self.available_users)
        )
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user] if data[1] >= 4][
            : self.state_size
        ]
        self.recommended_items = set(self.items)

        self.done = False
        self.done_count = done_count

        self.n_groups = n_groups
        self.movies_groups = movies_groups
        self.group_count = {k: 0 for k in range(1, self.n_groups + 1)}
        self.total_recommended_items = 0
        self.fairness_constraints = fairness_constraints

        self.reward_model = reward_model
        self.use_only_reward_model = use_only_reward_model

    def _generate_available_users(self):
        available_users = []
        for u in self.users_dict.keys():
            positive_items = [data[0] for data in self.users_dict[u] if data[1] >= 4]
            if len(positive_items) > self.state_size:
                available_users.append(u)
        return available_users

    def get_reward(self, action):
        reward = 0

        if not self.use_only_reward_model:

            # If we know the reward according to the feedback log, we use it
            if (
                action in self.user_items.keys()
                and action not in self.recommended_items
            ):
                if self.reward_model:
                    reward = 0.5 * (self.user_items[action] - 3)
                else:
                    reward = 1 if self.user_items[action] >= 4 else -1

            # If we dont know the reward, use reward predictor
            elif (
                action not in self.user_items.keys()
                and action not in self.recommended_items
            ):
                if self.reward_model:
                    reward = (
                        self.reward_model.predict(
                            torch.tensor([self.user]).long().to(self.device),
                            torch.tensor([action]).long().to(self.device),
                        )
                        .detach()
                        .cpu()
                        .numpy()[0]
                    )
                else:
                    reward = 0

            # Penalize the agent if they have already recommended this item
            else:
                reward = -1.5

        else:
            if action not in self.recommended_items:
                reward = (
                    self.reward_model.predict(
                        torch.tensor([self.user]).long().to(self.device),
                        torch.tensor([action]).long().to(self.device),
                    )
                    .detach()
                    .cpu()
                    .numpy()[0]
                )
            else:
                reward = -1.5

        return reward

    def reset(self):
        self.user = (
            self.fix_user_id
            if self.fix_user_id
            else np.random.choice(self.available_users)
        )
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user] if data[1] >= 4][
            : self.state_size
        ]
        self.done = False
        self.recommended_items = set(self.items)
        self.group_count = {k: 0 for k in range(1, self.n_groups + 1)}
        self.total_recommended_items = 0
        return self.user, self.items, self.done

    def step(self, action, top_k=False):

        if top_k:
            correctly_recommended, rewards = [], []
            for act in action:
                self.group_count[self.movies_groups[act]] += 1
                self.total_recommended_items += 1

                _reward = self.get_reward(act)
                rewards.append(_reward)

                if _reward > 0:
                    correctly_recommended.append(act)
                self.recommended_items.add(act)

            if max(rewards) > 0:
                self.items = (
                    self.items[len(correctly_recommended) :] + correctly_recommended
                )
                self.items = self.items[-self.state_size :]
            reward = rewards

        else:
            self.group_count[self.movies_groups[action]] += 1
            self.total_recommended_items += 1

            reward = self.get_reward(action)
            if reward > 0:
                self.items = self.items[1:] + [action]

            self.recommended_items.add(action)

        if (
            self.total_recommended_items > self.done_count
            or len(self.recommended_items)
            >= self.users_history_lens[list(self.users_dict.keys()).index(self.user)]
        ):
            self.done = True

        return self.items, reward, self.done, self.recommended_items


class OfflineFairEnv(OfflineEnv):
    def __init__(
        self,
        users_dict,
        users_history_lens,
        n_groups,
        movies_groups,
        state_size,
        done_count,
        fairness_constraints,
        fix_user_id=None,
        reward_model=None,
        use_only_reward_model=False,
        device="cpu",
    ):
        super().__init__(
            users_dict,
            users_history_lens,
            n_groups,
            movies_groups,
            state_size,
            done_count,
            fairness_constraints,
            fix_user_id,
            reward_model,
            use_only_reward_model,
            device,
        )

    def get_fair_reward(self, group):
        reward = self.fairness_constraints[group - 1] / np.sum(
            self.fairness_constraints
        )
        -(self.group_count[group] / np.sum(list(self.group_count.values())))
        +1

        return reward

    def step(self, action, top_k=False):

        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
                group = self.movies_groups[act]
                self.group_count[group] += 1
                self.total_recommended_items += 1

                _reward = self.get_reward(act)
                if _reward > 0:
                    correctly_recommended.append(act)
                    rewards.append(self.get_fair_reward(group))
                elif _reward == 0 or _reward == -1.5:
                    rewards.append(_reward)
                else:
                    rewards.append(-1)

                self.recommended_items.add(act)

            if max(rewards) > 0:
                self.items = (
                    self.items[len(correctly_recommended) :] + correctly_recommended
                )
                self.items = self.items[-self.state_size :]
            reward = rewards

        else:
            group = self.movies_groups[action]
            self.group_count[group] += 1
            self.total_recommended_items += 1

            _reward = self.get_reward(action)

            if _reward > 0:
                self.items = self.items[1:] + [action]
                reward = self.get_fair_reward(group)
            elif _reward == 0 or _reward == -1.5:
                reward = _reward
            else:
                reward = -1

            self.recommended_items.add(action)

        if (
            len(self.recommended_items) > self.done_count
            or len(self.recommended_items)
            >= self.users_history_lens[list(self.users_dict.keys()).index(self.user)]
        ):
            self.done = True

        return self.items, reward, self.done, self.recommended_items
