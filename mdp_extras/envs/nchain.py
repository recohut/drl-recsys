"""Utilities for working with the OpenAI Gym NChain MDP"""

import numpy as np

from mdp_extras import DiscreteExplicitExtras, Indicator, Linear


HUMAN_ACTIONS = ["→", "←"]


def nchain_extras(env, gamma=0.99):
    """Get extras for a gym.envs.toy_text.nchain.NChainEnv
    
    Args:
        env (gym.envs.toy_text.nchain.NChainEnv): Environment
        gamma (float): Discount factor
    
    Returns:
        (DiscreteExplicitExtras): Extras object
        (Indicator): State-action indicator feature function
        (Linear): Linear reward function
    """

    # How to handle <TimeLimit<______>> and other Wrappers?
    # assert isinstance(env, gym.envs.toy_text.nchain.NChainEnv)

    # Action constants
    A_FORWARD = 0
    A_BACKWARD = 1

    states = np.arange(env.observation_space.n)
    actions = np.arange(env.action_space.n)

    p0s = np.zeros(env.observation_space.n)
    p0s[0] = 1.0

    # Populate dynamics
    t_mat = np.zeros(
        (env.observation_space.n, env.action_space.n, env.observation_space.n)
    )

    # Backward action moves to 0th state if it doesn't fail, forward if it does
    t_mat[:, A_BACKWARD, 0] = 1.0 - env.slip
    for s1 in states:
        t_mat[s1, A_BACKWARD, min(s1 + 1, env.observation_space.n - 1)] = env.slip

    # Forward action moves to next state if it doesn't fail, 0th if it does
    for s1 in states:
        t_mat[s1, A_FORWARD, min(s1 + 1, env.observation_space.n - 1)] = 1.0 - env.slip
    t_mat[:, A_FORWARD, 0] = env.slip

    terminal_state_mask = np.zeros(env.observation_space.n)

    xtr = DiscreteExplicitExtras(
        states, actions, p0s, t_mat, terminal_state_mask, gamma=gamma,
    )

    phi = Indicator(Indicator.Type.OBSERVATION_ACTION, xtr)

    state_action_rewards = np.zeros((env.observation_space.n, env.action_space.n))
    state_action_rewards[:, A_BACKWARD] = env.small
    state_action_rewards[env.observation_space.n - 1, A_FORWARD] = env.large
    reward = Linear(state_action_rewards.flatten())

    return (xtr, phi, reward)
