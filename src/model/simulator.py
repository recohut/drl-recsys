from copy import deepcopy
from typing import Callable

import pickle
import numpy as np
from tqdm import tqdm

from obp.policy.policy_type import PolicyType
from obp.types import BanditFeedback, BanditPolicy
from obp.utils import check_bandit_feedback_inputs
from obp.utils import convert_to_action_dist


def run_bandit_simulation(
    bandit_feedback: BanditFeedback,
    policy: BanditPolicy,
    epochs: int,
):

    """Run an online bandit algorithm on the given logged bandit feedback data.
    Parameters
    ----------
    bandit_feedback: BanditFeedback
        Logged bandit feedback data used in offline bandit simulation.
    policy: BanditPolicy
        Online bandit policy evaluated in offline bandit simulation (i.e., evaluation policy).
    Returns
    --------
    action_dist: array-like, shape (n_rounds, n_actions, len_list)
        Action choice probabilities (can be deterministic).
    """
    for key_ in ["action", "position", "reward", "pscore", "context"]:
        if key_ not in bandit_feedback:
            raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")
    check_bandit_feedback_inputs(
        context=bandit_feedback["context"],
        action=bandit_feedback["action"],
        reward=bandit_feedback["reward"],
        position=bandit_feedback["position"],
        pscore=bandit_feedback["pscore"],
    )

    policy_ = policy
    dim_context = bandit_feedback["context"].shape[1]
    if bandit_feedback["position"] is None:
        bandit_feedback["position"] = np.zeros_like(
            bandit_feedback["action"], dtype=int
        )

    # Instantiate trackers
    aligned_time_steps = 0
    cumulative_rewards = 0

    cvr = []
    aligned_cvr = []
    group_count = []
    propfair = []
    ufg = []

    for _ in range(epochs):
        selected_actions_list = list()
        policy_.clear_group_count()

        for action_, reward_, position_, context_ in tqdm(
            zip(
                bandit_feedback["action"],
                bandit_feedback["reward"],
                bandit_feedback["position"],
                bandit_feedback["context"],
            ),
            total=bandit_feedback["n_rounds"],
        ):

            # select a list of actions
            if policy_.policy_type == PolicyType.CONTEXT_FREE:
                selected_actions = policy_.select_action()
            elif policy_.policy_type == PolicyType.CONTEXTUAL:
                selected_actions = policy_.select_action(
                    context_.reshape(1, dim_context)
                )
            action_match_ = action_ == selected_actions[position_]
            # update parameters of a bandit policy
            # only when selected actions&positions are equal to logged actions&positions
            if action_match_:
                if policy_.policy_type == PolicyType.CONTEXT_FREE:
                    policy_.update_params(action=action_, reward=reward_)
                elif policy_.policy_type == PolicyType.CONTEXTUAL:
                    policy_.update_params(
                        action=action_,
                        reward=reward_,
                        context=context_.reshape(1, dim_context),
                    )

                # For CTR calculation
                aligned_time_steps += 1
                cumulative_rewards += reward_
                aligned_cvr.append(cumulative_rewards / aligned_time_steps)

            selected_actions_list.append(selected_actions)

        _propfair = policy_.propfair
        _cvr = cumulative_rewards / bandit_feedback["n_rounds"]
        _ufg = policy_.propfair / max(
            1 - (cumulative_rewards / bandit_feedback["n_rounds"]), 0.01
        )

        propfair.append(_propfair)
        cvr.append(_cvr)
        ufg.append(_ufg)
        group_count.append(policy_.group_count)

    with open("model/{}.pkl".format(policy_.policy_name), "wb") as file:
        pickle.dump(policy_, file)

    action_dist = convert_to_action_dist(
        n_actions=bandit_feedback["action"].max() + 1,
        selected_actions=np.array(selected_actions_list),
    )
    return action_dist, aligned_cvr, cvr, propfair, ufg, group_count
