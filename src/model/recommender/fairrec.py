import torch
import numpy as np

from src.model.recommender.drr import DRRAgent


class FairRecAgent(DRRAgent):
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

        super().__init__(
            env=env,
            users_num=users_num,
            items_num=items_num,
            state_size=state_size,
            srm_size=srm_size,
            model_path=model_path,
            embedding_network_weights_path=embedding_network_weights_path,
            emb_model=emb_model,
            train_version=train_version,
            is_test=is_test,
            use_wandb=use_wandb,
            embedding_dim=embedding_dim,
            actor_hidden_dim=actor_hidden_dim,
            actor_learning_rate=actor_learning_rate,
            critic_hidden_dim=critic_hidden_dim,
            critic_learning_rate=critic_learning_rate,
            discount_factor=discount_factor,
            tau=tau,
            learning_starts=learning_starts,
            replay_memory_size=replay_memory_size,
            batch_size=batch_size,
            n_groups=n_groups,
            fairness_constraints=fairness_constraints,
            no_cuda=no_cuda,
        )

    def get_state(self, user_id, items_ids):
        items_eb = self.get_items_emb(items_ids)
        groups_eb = []
        for items in items_ids:
            groups_id = [
                k
                for k, v in self.env.movies_groups.items()
                if v == self.env.movies_groups[items]
            ]
            groups_eb.append(self.get_items_emb(groups_id))

        total_exp = np.sum(list(self.env.group_count.values()))
        fairness_allocation = (
            (np.array(list(self.env.group_count.values())) / total_exp)
            if total_exp > 0
            else np.zeros(self.n_groups)
        )

        with torch.no_grad():
            ## SRM state
            state = self.srm_ave(
                [
                    items_eb.unsqueeze(0),
                    groups_eb,
                    torch.FloatTensor(fairness_allocation).unsqueeze(0).to(self.device),
                ]
            )

        return state
