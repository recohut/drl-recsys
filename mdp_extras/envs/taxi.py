"""Utilities for working with the OpenAI Gym Taxi MDP"""

import gym
import numpy as np

from mdp_extras import DiscreteExplicitExtras, Indicator, Linear


human_actions = ["↓", "↑", "←", "→", "P", "D"]


def taxi_extras(env, gamma=0.99):
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
    # assert isinstance(env, gym.envs.toy_text.taxi.TaxiEnv)

    xtr = DiscreteExplicitExtras.fromdiscrete(env, gamma=gamma)

    # FrozenLake uses state-action-based indicator features
    phi = Indicator(Indicator.Type.OBSERVATION_ACTION, xtr)

    # Copy reward function by exhaustively testing the environment
    original_state = env.s
    theta = np.zeros((len(xtr.states), len(xtr.actions)))
    for s in xtr.states:
        for a in xtr.actions:
            env.reset()
            env.s = s
            _, theta[s, a], _, _ = env.step(a)
    env.s = original_state
    reward = Linear(theta.flatten())

    return xtr, phi, reward
