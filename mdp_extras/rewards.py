import abc

import numpy as np
import itertools as it

import mdp_extras.features as f


class RewardFunction(abc.ABC):
    """An abstract base class for reward functions"""

    def __init__(self):
        raise NotImplementedError

    def __call__(self, v):
        """Get reward given feature input

        Args:
            v (numpy array): Feature vector

        Returns:
            (float): Scalar reward
        """
        raise NotImplementedError


class Linear(RewardFunction):
    """A reward function linear in a feature function"""

    def __init__(self, theta):
        """C-tor

        Args:
            theta (numpy array): Linear reward weights
        """
        self.theta = theta

    def __call__(self, v):
        """Get reward given feature input

        Args:
            v (numpy array): Feature vector

        Returns:
            (float): Scalar reward
        """
        return self.theta @ v

    @property
    def range(self):
        """Get the range of values this LinearReward can take"""
        return (np.min(self.theta), np.max(self.theta))

    @staticmethod
    def fromdiscrete(env, xtr=None):
        """Builds LienarReward from DiscreteEnv

        Args:
            env (gym.envs.toy_text.discrete.DiscreteEnv): Environment to build Extras
                from

            xtr (DiscreteExplicitExtras): Optional extras object, if not provided, will
                be constructed locally

        Returns:
            (Indicator): Indicator feature function
            (LinearReward): Reward function
        """

        if xtr is None:
            xtr = f.DiscreteExplicitExtras.fromdiscrete(env)

        # Infer reward structure from transition tensor
        _rs = {s: set() for s in xtr.states}
        _rsa = {sa: set() for sa in it.product(xtr.states, xtr.actions)}
        _rsas = {sas: set() for sas in it.product(xtr.states, xtr.actions, xtr.states)}
        for s1 in xtr.states:

            if xtr.terminal_state_mask[s1]:
                # Don't consider transitions from terminal states as they are invalid
                continue

            for a in xtr.actions:
                for prob, s2, r, done in env.P[s1][a]:
                    _rs[s2].add(r)
                    _rsa[(s1, a)].add(r)
                    _rsas[(s1, a, s2)].add(r)
        _rs = {s: list(rewards) for s, rewards in _rs.items()}
        _rsa = {sa: list(rewards) for sa, rewards in _rsa.items()}
        _rsas = {sas: list(rewards) for sas, rewards in _rsas.items()}
        num_rewards_per_state = [len(rewards) for rewards in _rs.values()]
        num_rewards_per_state_action = [len(rewards) for rewards in _rsa.values()]
        num_rewards_per_state_action_state = [
            len(rewards) for rewards in _rsas.values()
        ]

        # Determine required feature function
        if max(num_rewards_per_state) == 1:

            # This MDP is consistent with state rewards
            phi = Indicator(Indicator.Type.OBSERVATION, xtr)
            theta = np.zeros(env.nS)
            for s in xtr.states:
                if len(_rs[s]) == 1:
                    theta[s] = _rs[s][0]
            r = Linear(theta.flatten())

        elif max(num_rewards_per_state_action) == 1:

            # This MDP is consistent with state-action rewards
            phi = Indicator(Indicator.Type.OBSERVATION_ACTION)
            theta = np.zeros((env.nS, env.nA))
            for s1, a in it.product(xtr.states, xtr.actions):
                if xtr.terminal_state_mask[s1]:
                    continue
                if len(_rsa[(s1, a)]) == 1:
                    theta[s1, a] = _rsa[(s1, a)][0]
            r = Linear(theta.flatten())

        elif max(num_rewards_per_state_action_state) == 1:

            # This MDP is consistent with state-action-state rewards
            phi = Indicator(Indicator.Type.OBSERVATION_ACTION_OBSERVATION)
            theta = np.zeros((env.nS, env.nA, env.nS))
            for s1, a, s2 in it.product(xtr.states, xtr.actions, xtr.states):
                if xtr.terminal_state_mask[s1]:
                    continue
                if len(_rsas[(s1, a, s2)]) == 1:
                    theta[s1, a, s2] = _rsas[(s1, a, s2)][0]
            r = Linear(theta.flatten())

        else:
            raise ValueError(
                "MDP rewards are stochastic and can't be represented by a linear reward function"
            )

        return phi, r

    def structured(self, xtr, phi):
        """Re-shape to structured reward representation matrices

        Args:
            xtr (DiscreteExplicitExtras): Extras object
            phi (Indicator): Indicator feature function for this reward
        """
        rs = np.zeros(len(xtr.states), dtype=np.float)
        rsa = np.zeros((len(xtr.states), len(xtr.actions)), dtype=np.float)
        rsas = np.zeros(
            (len(xtr.states), len(xtr.actions), len(xtr.states)), dtype=np.float
        )
        if phi.type == f.Indicator.Type.OBSERVATION:
            for o1 in xtr.states:
                rs[o1] = self(phi(o1))
        elif phi.type == f.Indicator.Type.OBSERVATION_ACTION:
            for o1 in xtr.states:
                for a in xtr.actions:
                    rsa[o1, a] = self(phi(o1, a))
        elif phi.type == f.Indicator.Type.OBSERVATION_ACTION_OBSERVATION:
            for o1 in xtr.states:
                for a in xtr.actions:
                    for o2 in xtr.states:
                        rsas[o1, a, o2] = self(phi(o1, a, o2))
        return rs, rsa, rsas
