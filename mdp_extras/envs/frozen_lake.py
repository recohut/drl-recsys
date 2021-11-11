"""Utilities for working with the OpenAI Gym FrozenLake MDP"""

import gym
import numpy as np

from mdp_extras import DiscreteExplicitExtras, Indicator, Linear


HUMAN_ACTIONS = ["←", "↓", "→", "↑"]


def frozen_lake_extras(env, gamma=0.99):
    """Get extras for a gym.envs.toy_text.frozen_lake.FrozenLakeEnv
    
    Args:
        env (gym.envs.toy_text.frozen_lake.FrozenLakeEnv): Environment
        gamma (float): Discount factor
    
    Returns:
        (DiscreteExplicitExtras): Extras object
        (Indicator): State-action indicator feature function
        (Linear): Linear reward function
    """

    # How to handle <TimeLimit<______>> and other Wrappers?
    # assert isinstance(env, gym.envs.toy_text.frozen_lake.FrozenLakeEnv)

    xtr = DiscreteExplicitExtras.fromdiscrete(env, gamma=gamma)

    # FrozenLake uses state-based indicator features
    phi = Indicator(Indicator.Type.OBSERVATION, xtr)

    # FrozenLake - agent gets low reward (0) for all states except the goal (final
    # state), where it gets the high reward (1).
    theta = np.zeros(len(phi)) + env.reward_range[0]
    theta[-1] = env.reward_range[1]
    reward = Linear(theta)

    return xtr, phi, reward
