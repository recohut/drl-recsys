import abc
import copy
import warnings
from enum import Enum

import numpy as np

import mdp_extras.utils as u
import mdp_extras.rewards as r


class FeatureFunction(abc.ABC):
    """Abstract feature function base class"""

    class Type(Enum):
        """Simple enum for input type to a feature/reward function"""

        OBSERVATION = "observation"
        OBSERVATION_ACTION = "observation_action"
        OBSERVATION_ACTION_OBSERVATION = "observation_action_observation"

        def check_args(self, o1, a, o2):
            """Checks that the given set of args are sufficient for this input type"""
            if self.has_o1:
                assert o1 is not None

            if self.has_a:
                assert a is not None

            if self.has_o2:
                assert o2 is not None

        @property
        def has_o1(self):
            return (
                (self == FeatureFunction.Type.OBSERVATION)
                or (self == FeatureFunction.Type.OBSERVATION_ACTION)
                or (self == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION)
            )

        @property
        def has_a(self):
            return (self == FeatureFunction.Type.OBSERVATION_ACTION) or (
                self == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION
            )

        @property
        def has_o2(self):
            return self == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION

    def __init__(self, type):
        """C-tor

        Args:
            type (InputType): Type of indicator features to construct
        """
        self.type = type

    def __len__(self):
        """Get length of the feature vector

        Returns:
            (int): Length of this feature vector
        """
        raise NotImplementedError

    def __call__(self, o1, a=None, o2=None):
        """Get feature vector given state(s) and/or action

        Args:
            o1 (any): Current observation
            a (any): Current action
            o2 (any): Next observation

        Returns:
            (numpy array): Feature vector for the current state(s) and/or action
        """
        raise NotImplementedError

    def onpath(self, rollout, gamma=1.0):
        """Compute discounted feature vector sum over an entire trajectory

        Args:
            rollout (list): (s, a) rollout

            gamma (float): Discount factor to use

        Returns:
            (numpy array): Discounted feature vector over the entire path
        """
        phi = np.zeros(len(self))
        if self.type == self.Type.OBSERVATION:
            for t, (o1, _) in enumerate(rollout):
                phi += self(o1) * (gamma ** t)
        elif self.type == self.Type.OBSERVATION_ACTION:
            for t, (o1, a) in enumerate(rollout[:-1]):
                phi += self(o1, a) * (gamma ** t)
        elif self.type == self.Type.OBSERVATION_ACTION_OBSERVATION:
            for t, (o1, a) in enumerate(rollout[:-1]):
                o2 = rollout[t + 1][0]
                phi += self(o1, a, o2) * (gamma ** t)
        else:
            raise ValueError

        return phi

    def demo_average(self, rollouts, gamma=1.0, weights=None):
        """Get empirical discounted feature expectation for a collection of rollouts

        Args:
            rollouts (list): List of (s, a) rollouts to average over

            gamma (float): Discount factor to use
            weights (numpy array): Optional weights to use for weighted feature vector averaging

        Returns:
            (numpy array): Average discounted feature vector over all paths
                in the dataset
        """

        # Catch case when only a single rollout is passed
        if isinstance(rollouts[0], tuple):
            rollouts = [rollouts]

        if weights is None:
            # Default to uniform path weighting
            weights = np.ones(len(rollouts)) / len(rollouts)
        else:
            assert len(weights) == len(
                rollouts
            ), f"Path weights are not correct size, should be {len(rollouts)}, are {len(weights)}"
            if not np.isclose(np.sum(weights), 1.0):
                warnings.warn(
                    f"Computing feature expectation with non-normalized weights (sum is {np.sum(weights)}) - did you mean to do this?"
                )

        phi_bar = np.zeros(len(self))
        for rollout, weight in zip(rollouts, weights):
            phi_bar += weight * self.onpath(rollout, gamma)

        # Apply normalization
        phi_bar /= np.sum(weights)

        return phi_bar

    def feature_distance(self, demo_1, demo_2, gamma=1.0):
        """Compute feature distance between two demonstrations

        This can be used as a cheap symmetric metric for preference matching between
        two paths

        Args:
            demo_1 (list): Path 1 (list of (s, a))
            demo_2 (list): Path 2 (list of (s, a))
            gamma (float): Discount factor

        Returns:
            (float): Feature distance, same units as the feature function
        """
        phi_bar_1 = self.onpath(demo_1, gamma=gamma)
        phi_bar_2 = self.onpath(demo_2, gamma=gamma)
        return np.linalg.norm(phi_bar_1 - phi_bar_2)


class Indicator(FeatureFunction):
    """Indicator feature function"""

    def __init__(self, type, xtr):
        """C-tor

        Args:
            type (InputType): Type of indicator features to construct
            xtr (DiscreteExplicitExtras): MDP definition
        """
        super().__init__(type)

        if self.type == FeatureFunction.Type.OBSERVATION:
            self._vec = np.zeros(len(xtr.states))
        elif self.type == FeatureFunction.Type.OBSERVATION_ACTION:
            self._vec = np.zeros((len(xtr.states), len(xtr.actions)))
        elif self.type == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION:
            self._vec = np.zeros((len(xtr.states), len(xtr.actions), len(xtr.sates)))
        else:
            raise ValueError

    def __len__(self):
        """Get length of the feature vector

        Returns:
            (int): Length of this feature vector
        """
        return len(self._vec.flatten())

    def __call__(self, o1, a=None, o2=None):
        """Get feature vector given state(s) and/or action

        Args:
            o1 (any): Current observation
            a (any): Current action
            o2 (any): Next observation

        Returns:
            (numpy array): Feature vector for the current state(s) and/or action
        """
        self.type.check_args(o1, a, o2)

        self._vec.flat[:] = 0
        try:
            if self.type == FeatureFunction.Type.OBSERVATION:
                self._vec[o1] = 1.0
            elif self.type == FeatureFunction.Type.OBSERVATION_ACTION:
                self._vec[o1, a] = 1.0
            elif self.type == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION:
                self._vec[o1, a, o2] = 1.0
            else:
                raise ValueError
        except IndexError:
            warnings.warn(
                f"Requested φ({o1}, {a}, {o2}), however slice is out-of-bounds. This could be due to using padded rollouts, in which case you can safely ignore this warning.",
                u.PaddedMDPWarning,
            )
        return self._vec.flatten().copy()


class Disjoint(FeatureFunction):
    """A feature function where each input has one of a set of disjoint values"""

    def __init__(self, type, xtr, values):
        """C-tor

        Args:
            type (InputType): Type of indicator features to construct
            xtr (DiscreteExplicitExtras): MDP definition
            values (numpy array): Vector of integer feature values, one for each
                state/state-action/state-action-state tuple, depending on type.
        """
        super().__init__(type)

        # How many feature values do we have?
        self._len = np.max(values) + 1

        self._vec = np.zeros(self._len)

        if self.type == FeatureFunction.Type.OBSERVATION:
            self._values = values.reshape((len(xtr.states)))
        elif self.type == FeatureFunction.Type.OBSERVATION_ACTION:
            self._values = values.reshape((len(xtr.states), len(xtr.actions)))
        elif self.type == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION:
            self._values = values.reshape(
                (len(xtr.states), len(xtr.actions), len(xtr.states))
            )
        else:
            raise ValueError

    def __len__(self):
        return self._len

    def __call__(self, o1, a=None, o2=None):
        """Get feature vector given state(s) and/or action

        Args:
            o1 (any): Current observation
            a (any): Current action
            o2 (any): Next observation

        Returns:
            (numpy array): Feature vector for the current state(s) and/or action
        """
        self.type.check_args(o1, a, o2)

        self._vec[:] = 0
        try:
            if self.type == FeatureFunction.Type.OBSERVATION:
                self._vec[self._values[o1]] = 1.0
            elif self.type == FeatureFunction.Type.OBSERVATION_ACTION:
                self._vec[self._values[o1, a]] = 1.0
            elif self.type == FeatureFunction.Type.OBSERVATION_ACTION_OBSERVATION:
                self._vec[self._values[o1, a, o2]] = 1.0
            else:
                raise ValueError
        except IndexError:
            warnings.warn(
                f"Requested φ({o1}, {a}, {o2}), however slice is out-of-bounds. This could be due to using padded rollouts, in which case you can safely ignore this warning.",
                u.PaddedMDPWarning,
            )
        return self._vec.copy()


class MirrorWrap(FeatureFunction):
    """A class that extends another feature function with it's negation, and the constant zero

    This procedure is useful for IRL algorithms that only support convex reward weights on a simplex, and is
    described in detail in section 5.2, "Representation Error" of

     * Syed, Umar, and Robert E. Schapire. "A game-theoretic approach to apprenticeship learning." Advances in neural
       information processing systems. 2008.

    """

    def __init__(self, inner_class):
        """C-tor

        Args:
            inner_class (FeatureFunction): FeatureFunction that we are mirroring here
        """
        self.inner_class = inner_class

    @property
    def type(self):
        """Reflect the inner class' type"""
        return self.inner_class.type

    def __len__(self):
        """Return the length of the wrapped feature vec

        Returns:
            (int): Length of this feature vector
        """
        # We add the negation (x2) and the constant zero (+1)
        return self.inner_class.__len__() * 2 + 1

    def __call__(self, o1, a=None, o2=None):
        """Get the wrapped feature vector given state(s) and/or action

        Args:
            o1 (any): Current observation
            a (any): Current action
            o2 (any): Next observation

        Returns:
            (numpy array): Feature vector for the current state(s) and/or action
        """
        feature_vec = self.inner_class(o1, a, o2)
        negation = np.array(-1.0 * feature_vec)
        zero_constant = np.array([0.0])
        return np.concatenate((feature_vec, negation, zero_constant))

    def update_reward(self, reward):
        """Update a reward function object to match the new feature function"""
        if not isinstance(reward, r.Linear):
            raise NotImplementedError

        assert len(reward.theta) == len(self.inner_class)

        new_theta = np.zeros(len(self))
        new_theta[: len(self.inner_class)] = reward.theta
        return r.Linear(new_theta)

    def unupdate_reward(self, reward):
        """Un-Update a reward function object to match the original feature function"""
        if not isinstance(reward, r.Linear):
            raise NotImplementedError

        assert len(reward.theta) == len(self)

        new_theta = np.zeros(len(self.inner_class))

        # The first half of the features get added
        new_theta += reward.theta[0 : len(new_theta)]

        # The second half of the features get subtracted
        new_theta -= reward.theta[len(new_theta) : -1]

        # The final 'zero' element does nothing

        return r.Linear(new_theta)
