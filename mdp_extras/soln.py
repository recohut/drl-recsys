"""Find solutions to MDPs via value iteration methods"""

import abc
import copy
import pickle
import torch
import warnings

import numpy as np
import torch.nn as nn
import itertools as it


from numba import jit

from mdp_extras.features import FeatureFunction
from mdp_extras.rewards import Linear
from mdp_extras.utils import (
    DiscreteExplicitLinearEnv,
    PaddedMDPWarning,
    softmax,
    mellowmax,
)


def vi(
    xtr,
    phi,
    reward,
    eps=1e-6,
    verbose=False,
    max_iter=None,
    type="rational",
    temperature=1.0,
):
    """Value iteration for the optimal state and state-action value function

        In our definition of terminal states, the agent receives a state reward upon reaching a terminal state,
        and *then* the episode ends. For this reason, the state-value of a terminal state is equal to the reward of that
        state, not 0 (as in Sutton and Barto). Likewise, the state-action value of a terminal state is equal to the
        reward of that state, and does not include the reward for executing an action in that state (because the agent
        doesn't get to execute an action in that state).

    Args:
        xtr (DiscreteExplicitExtras): Extras object
        phi (FeatureFunction): Feature function
        reward (RewardFunction): Reward function

        eps (float): Value convergence tolerance
        verbose (bool): Extra logging
        max_iter (int): If provided, iteration will terminate regardless of convergence
            after this many iterations.
        type (str): Type of value functions to calculate. Options include
             - 'rational' - Regular VI - take max over future actions
             - 'soft' - Soft VI - take softmax over future actions
             - 'mellow' - Mellow VI - take mellowmax over future actions
            N.b. - for now, softmax and mellowmax do not make use of Numba JIT optimization
        temperature (float): Temperature parameter, only used for soft or mellomax. As temp -> 0, these
            VI algorithms approach rational VI.

    Returns:
        (numpy array): |S| array of state values
        (numpy array): |S|x|A| matrix of state-action values
    """

    assert type in ("rational", "soft", "mellow"), f"Unknown type parameter {type}"

    @jit(nopython=True)
    def nb_vi_rational(
        t_mat,
        terminal_states,
        gamma,
        rs,
        rsa,
        rsas,
        eps=1e-6,
        verbose=False,
        max_iter=None,
    ):
        """Value iteration to find the optimal rational value function

        Args:
            t_mat (numpy array): |S|x|A|x|S| transition matrix
            terminal_states (numpy array): |S| boolean array of terminal state indicators
            gamma (float): Discount factor
            rs (numpy array): |S| State reward vector
            rsa (numpy array): |S|x|A| State-action reward vector
            rsas (numpy array): |S|x|A|x|S| State-action-state reward vector

            eps (float): Value convergence tolerance
            verbose (bool): Extra logging
            max_iter (int): If provided, iteration will terminate regardless of convergence
                after this many iterations.

        Returns:
            (numpy array): |S| vector of optimal rational state values
            (numpy array): |S|x|A| matrix of optimal rational state-action values
        """

        s_value = np.zeros(t_mat.shape[0])
        sa_value = np.zeros((t_mat.shape[0], t_mat.shape[1]))
        for s1 in range(t_mat.shape[0]):
            if terminal_states[s1]:
                s_value[s1] = rs[s1]
                sa_value[s1, :] = rs[s1]

        _iter = 0
        while True:
            delta = 0
            for s1 in range(t_mat.shape[0]):
                if terminal_states[s1]:
                    # Don't update terminal states
                    continue
                v = s_value[s1]
                sa_value[s1, :] = np.zeros(t_mat.shape[1])
                for a in range(t_mat.shape[1]):
                    for s2 in range(t_mat.shape[2]):
                        sa_value[s1, a] += t_mat[s1, a, s2] * (
                            rs[s1] + rsa[s1, a] + rsas[s1, a, s2] + gamma * s_value[s2]
                        )
                s_value[s1] = np.max(sa_value[s1, :])
                delta = max(delta, np.abs(v - s_value[s1]))

            if max_iter is not None and _iter >= max_iter:
                if verbose:
                    print("Terminating before convergence, # iterations = ", _iter)
                    break

            # Check value function convergence
            if delta < eps:
                if verbose:
                    print("Value Iteration #", _iter, " delta=", delta, " (converged)")
                break
            else:
                if verbose:
                    print("Value Iteration #", _iter, " delta=", delta)

            _iter += 1

        return s_value, sa_value

    # @jit(nopython=True)
    def nb_vi_soft(
        t_mat,
        terminal_states,
        gamma,
        rs,
        rsa,
        rsas,
        temperature,
        eps=1e-6,
        verbose=False,
        max_iter=None,
    ):
        """Value iteration to find the soft optimal value functions

        Args:
            t_mat (numpy array): |S|x|A|x|S| transition matrix
            terminal_states (numpy array): |S| boolean array of terminal state indicators
            gamma (float): Discount factor
            rs (numpy array): |S| State reward vector
            rsa (numpy array): |S|x|A| State-action reward vector
            rsas (numpy array): |S|x|A|x|S| State-action-state reward vector
            temperature (float): Temperature for softmax

            eps (float): Value convergence tolerance
            verbose (bool): Extra logging
            max_iter (int): If provided, iteration will terminate regardless of convergence
                after this many iterations.

        Returns:
            (numpy array): |S| vector of optimal soft state values
            (numpy array): |S|x|A| matrix of optimal soft state-action values
        """

        s_value = np.zeros(t_mat.shape[0])
        sa_value = np.zeros((t_mat.shape[0], t_mat.shape[1]))
        for s1 in range(t_mat.shape[0]):
            if terminal_states[s1]:
                s_value[s1] = rs[s1]
                sa_value[s1, :] = rs[s1]

        _iter = 0
        while True:
            delta = 0
            for s1 in range(t_mat.shape[0]):
                if terminal_states[s1]:
                    # Don't update terminal states
                    continue
                v = s_value[s1]
                sa_value[s1, :] = np.zeros(t_mat.shape[1])
                for a in range(t_mat.shape[1]):
                    for s2 in range(t_mat.shape[2]):
                        sa_value[s1, a] += t_mat[s1, a, s2] * (
                            rs[s1] + rsa[s1, a] + rsas[s1, a, s2] + gamma * s_value[s2]
                        )
                s_value[s1] = softmax(sa_value[s1, :], temperature)
                delta = max(delta, np.abs(v - s_value[s1]))

            if max_iter is not None and _iter >= max_iter:
                if verbose:
                    print("Terminating before convergence, # iterations = ", _iter)
                    break

            # Check value function convergence
            if delta < eps:
                if verbose:
                    print("Value Iteration #", _iter, " delta=", delta, " (converged)")
                break
            else:
                if verbose:
                    print("Value Iteration #", _iter, " delta=", delta)

            _iter += 1

        return s_value, sa_value

    # @jit(nopython=True)
    def nb_vi_mellow(
        t_mat,
        terminal_states,
        gamma,
        rs,
        rsa,
        rsas,
        temperature,
        eps=1e-6,
        verbose=False,
        max_iter=None,
    ):
        """Value iteration to find the mellow optimal value functions

        Args:
            t_mat (numpy array): |S|x|A|x|S| transition matrix
            terminal_states (numpy array): |S| boolean array of terminal state indicators
            gamma (float): Discount factor
            rs (numpy array): |S| State reward vector
            rsa (numpy array): |S|x|A| State-action reward vector
            rsas (numpy array): |S|x|A|x|S| State-action-state reward vector
            temperature (float): Temperature for mellowmax

            eps (float): Value convergence tolerance
            verbose (bool): Extra logging
            max_iter (int): If provided, iteration will terminate regardless of convergence
                after this many iterations.

        Returns:
            (numpy array): |S| vector of optimal mellow state values
            (numpy array): |S|x|A| matrix of optimal mellow state-action values
        """

        s_value = np.zeros(t_mat.shape[0])
        sa_value = np.zeros((t_mat.shape[0], t_mat.shape[1]))
        for s1 in range(t_mat.shape[0]):
            if terminal_states[s1]:
                s_value[s1] = rs[s1]
                sa_value[s1, :] = rs[s1]

        _iter = 0
        while True:
            delta = 0
            for s1 in range(t_mat.shape[0]):
                if terminal_states[s1]:
                    # Don't update terminal states
                    continue
                v = s_value[s1]
                sa_value[s1, :] = np.zeros(t_mat.shape[1])
                for a in range(t_mat.shape[1]):
                    for s2 in range(t_mat.shape[2]):
                        sa_value[s1, a] += t_mat[s1, a, s2] * (
                            rs[s1] + rsa[s1, a] + rsas[s1, a, s2] + gamma * s_value[s2]
                        )
                s_value[s1] = mellowmax(sa_value[s1, :], temperature)
                delta = max(delta, np.abs(v - s_value[s1]))

            if max_iter is not None and _iter >= max_iter:
                if verbose:
                    print("Terminating before convergence, # iterations = ", _iter)
                    break

            # Check value function convergence
            if delta < eps:
                if verbose:
                    print("Value Iteration #", _iter, " delta=", delta, " (conveged)")
                break
            else:
                if verbose:
                    print("Value Iteration #", _iter, " delta=", delta)

            _iter += 1

        return s_value, sa_value

    xtr = xtr.as_unpadded
    if type == "rational":
        return nb_vi_rational(
            xtr.t_mat,
            np.array(xtr.terminal_state_mask),
            xtr.gamma,
            *reward.structured(xtr, phi),
            eps=eps,
            verbose=verbose,
            max_iter=max_iter,
        )
    elif type == "soft":
        return nb_vi_soft(
            xtr.t_mat,
            np.array(xtr.terminal_state_mask),
            xtr.gamma,
            *reward.structured(xtr, phi),
            temperature,
            eps=eps,
            verbose=verbose,
            max_iter=max_iter,
        )
    elif type == "mellow":
        return nb_vi_mellow(
            xtr.t_mat,
            np.array(xtr.terminal_state_mask),
            xtr.gamma,
            *reward.structured(xtr, phi),
            temperature,
            eps=eps,
            verbose=verbose,
            max_iter=max_iter,
        )
    else:
        raise ValueError(f"Unknown type parameter {type}")


def q2v(q_star):
    """Convert optimal rational state-action value function to optimal state value function

    Args:
        q_star (numpy array): |S|x|A| Optimal state-action value function array

    Returns:
        (numpy array): |S| Optimal state value function vector
    """
    return np.max(q_star, axis=1)


def v2q(v_star, xtr, phi, reward):
    """Convert optimal state value function to optimal state-action value function

    Args:
        v_star (numpy array): |S| Optimal state value function vector
        xtr (DiscreteExplicitExtras):
        phi (FeatureFunction): Feature function
        reward (RewardFunction): Reward function

    Returns:
        (numpy array): |S|x|A| Optimal state-action value function array
    """

    @jit(nopython=True)
    def _nb_q_from_v(
        v_star,
        terminal_states,
        t_mat,
        gamma,
        state_rewards,
        state_action_rewards,
        state_action_state_rewards,
    ):
        """Find Q* given V* (numba optimized version)

        Args:
            v_star (numpy array): |S| vector of optimal state values
            t_mat (numpy array): |S|x|A|x|S| transition matrix
            terminal_states (numpy array): |S| boolean indicator vector showing if a state is terminal
            gamma (float): Discount factor
            state_rewards (numpy array): |S| array of state rewards
            state_action_rewards (numpy array): |S|x|A| array of state-action rewards
            state_action_state_rewards (numpy array): |S|x|A|x|S| array of state-action-state rewards

        Returns:
            (numpy array): |S|x|A| array of optimal state-action values
        """

        q_star = np.zeros(t_mat.shape[0 : 1 + 1])
        for s1 in range(t_mat.shape[0]):
            if terminal_states[s1]:
                q_star[s1, :] = state_rewards[s1]

        for s1 in range(t_mat.shape[0]):
            if terminal_states[s1]:
                # Don't update terminal states
                continue
            for a in range(t_mat.shape[1]):
                for s2 in range(t_mat.shape[2]):
                    q_star[s1, a] += t_mat[s1, a, s2] * (
                        state_action_rewards[s1, a]
                        + state_action_state_rewards[s1, a, s2]
                        + state_rewards[s2]
                        + gamma * v_star[s2]
                    )

        return q_star

    xtr = xtr.as_unpadded
    return _nb_q_from_v(v_star, xtr.t_mat, xtr.gamma, *reward.structured(xtr, phi))


def pi_eval(xtr, phi, reward, policy, eps=1e-6, num_runs=1):
    """Determine the value function of a given policy

    Args:
        xtr (DiscreteExplicitExtras): Extras object
        phi (FeatureFunction): Feature function
        reward (RewardFunction): Reward function
        policy (object): Policy object providing a .predict(s) method to match the
            stable-baselines policy API

        eps (float): State value convergence threshold
        num_runs (int): Number of policy evaluations to average over - for deterministic
            policies, leave this as 1, but for stochastic policies, set to a large
            number (the function will then sample actions stochastically from the
            policy).

    Returns:
        (numpy array): |S| state value vector
    """

    @jit(nopython=True)
    def _nb_policy_evaluation(
        t_mat,
        terminal_states,
        gamma,
        rs,
        rsa,
        rsas,
        policy_vector,
        eps=1e-6,
    ):
        """Determine the value function of a given deterministic policy

        Args:
            t_mat (numpy array): |S|x|A|x|S| transition matrix
            terminal_states (numpy array): |S| boolean array of terminal state indicators
            gamma (float): Discount factor
            rs (numpy array): |S| State reward vector
            rsa (numpy array): |S|x|A| State-action reward vector
            rsas (numpy array): |S|x|A|x|S| State-action-state reward vector
            policy_vector (numpy array): |S| vector indicating action to take from each
                state

            eps (float): State value convergence threshold

        Returns:
            (numpy array): |S| state value vector
        """

        v_pi = np.zeros(t_mat.shape[0])
        for s1 in range(t_mat.shape[0]):
            if terminal_states[s1]:
                v_pi[s1] = rs[s1]

        _iteration = 0
        while True:
            delta = 0
            for s1 in range(t_mat.shape[0]):
                if terminal_states[s1]:
                    # Don't update terminal states
                    continue
                v = v_pi[s1]
                _tmp = 0
                for a in range(t_mat.shape[1]):
                    if policy_vector[s1] != a:
                        continue
                    for s2 in range(t_mat.shape[2]):
                        _tmp += t_mat[s1, a, s2] * (
                            rs[s1] + rsa[s1, a] + rsas[s1, a, s2] + gamma * v_pi[s2]
                        )
                v_pi[s1] = _tmp
                delta = max(delta, np.abs(v - v_pi[s1]))

            if delta < eps:
                break
            _iteration += 1

        return v_pi

    xtr = xtr.as_unpadded
    policy_state_values = []
    for _ in range(num_runs):
        action_vector = np.array([policy.predict(s)[0] for s in xtr.states])
        policy_state_values.append(
            _nb_policy_evaluation(
                xtr.t_mat,
                xtr.terminal_state_mask,
                xtr.gamma,
                *reward.structured(xtr, phi),
                action_vector,
                eps=eps,
            )
        )

    # Average over runs
    return np.mean(policy_state_values, axis=0)


def q_grad_fpi(theta, xtr, phi, tol=1e-3):
    """Estimate the Q-gradient with a Fixed Point Iteration

    TODO ajs 07/dec/2020 Handle state-action-state feature functions?

    This method uses a Fixed-Point estimate by Neu and Szepesvari 2007, and is
    considered by me to be the 'gold standard' for Q-gradient estimation.

    See "Apprenticeship learning using inverse reinforcement learning and gradient
    methods." by Neu and Szepesvari in UAI, 2007.

    This method requires |S|x|S|x|A|x|A| updates per iteration, and empirically appears
    to have exponential convergence in the number of iterations. That is,
    δ α O(exp(-1.0 x iteration)).

    Args:
        theta (numpy array): Current reward parameters
        xtr (mdp_extras.DiscreteExplicitExtras): MDP definition
        phi (mdp_extras.FeatureFunction): A state-action feature function

    Returns:
        (numpy array): |S|x|A|x|φ| Array of partial derivatives δQ(s, a)/dθ
    """

    assert (
        phi.type == phi.Type.OBSERVATION or phi.type == phi.Type.OBSERVATION_ACTION
    ), "Currently, state-action-state features are not supported with this method"

    xtr = xtr.as_unpadded

    # Get optimal *DETERMINISTIC* policy
    # (the fixed point iteration is only valid for deterministic policies)
    reward = Linear(theta)
    _, q_star = vi(xtr, phi, reward)
    pi_star = OptimalPolicy(q_star, stochastic=False)

    @jit(nopython=True)
    def _nb_fpi(states, actions, t_mat, gamma, phi, pi_star, tol):
        """Plain-object core loop for numba optimization

        TODO ajs 07/dec/2020 Handle state-action-state feature functions?

        Args:
            states (list): States
            actions (list): Actions
            t_mat (numpy array): |S|x|A|x|S| transition matrix
            gamma (float): Discount factor
            phi (numpy array): |S|x|A|x|φ| state-action feature matrix
            pi_star (numpy array): |S|x|A| policy matrix
            tol (float): Convergence threshold

        Returns:
            |S|x|A|x|φ| Estimate of gradient of Q function
        """
        # Initialize
        dq_dtheta = phi.copy()

        # Apply fixed point iteration
        it = 0
        while True:
            # Use full-width backups
            dq_dtheta_old = dq_dtheta.copy()
            dq_dtheta[:, :, :] = 0.0
            for s1 in states:
                for a1 in actions:
                    dq_dtheta[s1, a1, :] = phi[s1, a1, :]
                    for s2 in states:
                        for a2 in actions:
                            dq_dtheta[s1, a1, :] += (
                                gamma
                                * t_mat[s1, a1, s2]
                                * pi_star[s2, a2]
                                * dq_dtheta_old[s2, a2, :]
                            )

            delta = np.max(np.abs(dq_dtheta_old.flatten() - dq_dtheta.flatten()))
            it += 1

            if delta <= tol:
                break

        return dq_dtheta

    # Build plain object arrays
    _pi_star = np.zeros((len(xtr.states), len(xtr.actions)))
    _phi = np.zeros((len(xtr.states), len(xtr.actions), len(phi)))
    for s in xtr.states:
        for a in xtr.actions:
            _pi_star[s, a] = pi_star.prob_for_state_action(s, a)
            _phi[s, a, :] = phi(s, a)

    dq_dtheta = _nb_fpi(
        xtr.states, xtr.actions, xtr.t_mat, xtr.gamma, _phi, _pi_star, tol
    )

    return dq_dtheta


def q_grad_sim(
    theta,
    xtr,
    phi,
    max_rollout_length,
    rollouts_per_sa=100,
):
    """Estimate the Q-gradient with simulation

    This method samples many rollouts from the optimal stationary stochastic policy for
    every possible (s, a) pair. This can give arbitrarily bad gradient estimates
    when used with non-episodic MDPs due to the early truncation of rollouts.
    This method also gives arbitrarily bad gradient estimates for terminal states in
    episodic MDPs, unless the max rollout length is set sufficiently high.

    This method requires sampling |S|x|A|x(rollouts_per_sa) rollouts from the MDP,
    and is by far the slowest of the Q-gradient estimators.

    Args:
        theta (numpy array): Current reward parameters
        xtr (mdp_extras.DiscreteExplicitExtras): MDP definition
        phi (mdp_extras.FeatureFunction): A state-action feature function
        max_rollout_length (int): Maximum rollout length - this value is rather
            arbitrary, but must be set to a large value to give accurate estimates.

        rollouts_per_sa (int): Number of rollouts to sample for each (s, a) pair. If
            the environment has deterministic dynamics, it's OK to set this to a small
            number (i.e. 1).

    Returns:
        (numpy array): |S|x|A|x|φ| Array of partial derivatives δQ(s, a)/dθ
    """

    xtr = xtr.as_unpadded

    # Get optimal policy
    reward = Linear(theta)
    _, q_star = vi(xtr, phi, reward)
    pi_star = OptimalPolicy(q_star, stochastic=True)

    # Duplicate the MDP, but clear all terminal states
    xtr = copy.deepcopy(xtr)
    xtr._terminal_state_mask[:] = False
    env = DiscreteExplicitLinearEnv(xtr, phi, reward)

    # Calculate expected feature vector under pi for all starting state-action pairs
    dq_dtheta = np.zeros((len(xtr.states), len(xtr.actions), len(phi)))
    for s in xtr.states:
        for a in xtr.actions:
            # Start with desired state, action
            rollouts = pi_star.get_rollouts(
                env,
                rollouts_per_sa,
                max_path_length=max_rollout_length,
                start_state=s,
                start_action=a,
            )
            phi_bar = phi.demo_average(rollouts, gamma=xtr.gamma)
            dq_dtheta[s, a, :] = phi_bar

    return dq_dtheta


def q_grad_nd(theta, xtr, phi, dtheta=0.01):
    """Estimate the Q-gradient with 2-point numerical differencing

    This method requires solving for 2x|φ| Q* functions

    Args:
        theta (numpy array): Current reward parameters
        xtr (mdp_extras.DiscreteExplicitExtras): MDP definition
        phi (mdp_extras.FeatureFunction): A state-action feature function

        dtheta (float): Amount to increment reward parameters by

    Returns:
        (numpy array): |S|x|A|x|φ| Array of partial derivatives δQ(s, a)/dθ
    """

    xtr = xtr.as_unpadded

    # Calculate expected feature vector under pi for all starting state-action pairs
    dq_dtheta = np.zeros((len(xtr.states), len(xtr.actions), len(phi)))

    # Sweep linear reward parameters
    for theta_i in range(len(theta)):
        # Solve for Q-function with upper and lower reward parameter increments
        theta_lower = theta.copy()
        theta_lower[theta_i] -= dtheta
        _, q_star_lower = vi(xtr, phi, Linear(theta_lower))

        theta_upper = theta.copy()
        theta_upper[theta_i] += dtheta
        _, q_star_upper = vi(xtr, phi, Linear(theta_upper))

        # Take numerical difference to estimate gradient
        dq_dtheta[:, :, theta_i] = (q_star_upper - q_star_lower) / (2.0 * dtheta)

    return dq_dtheta


class Policy(abc.ABC):
    """A simple Policy base class

    Provides a .predict(s) method to match the stable-baselines policy API
    """

    def __init__(self):
        """C-tor"""
        raise NotImplementedError

    def save(self, path):
        """Save policy to file"""
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        """Load policy from file"""
        with open(path, "rb") as file:
            _self = pickle.load(file)
            return _self

    def predict(self, s):
        """Predict next action and distribution over states

        N.b. This function matches the API of the stabe-baselines policies.

        Args:
            s (int): Current state

        Returns:
            (int): Sampled action
            (None): Placeholder to ensure this function matches the stable-baselines
                policy interface. Some policies use this argument to return a prediction
                over future state distributions - we do not.
        """
        action = np.random.choice(np.arange(self.q.shape[1]), p=self.prob_for_state(s))
        return action, None

    def log_prob_for_state(self, s):
        """Get the action log probability vector for the given state

        Args:
            s (int): Current state

        Returns:
            (numpy array): Log probability distribution over actions
        """
        raise NotImplementedError

    def prob_for_state(self, s):
        """Get the action probability vector for the given state

        Args:
            s (int): Current state

        Returns:
            (numpy array): Probability distribution over actions
        """
        return np.exp(self.log_prob_for_state(s))

    def prob_for_state_action(self, s, a):
        """Get the probability for the given state, action

        Args:
            s (int): Current state
            a (int): Chosen action

        Returns:
            (float): Probability of choosing a from s
        """
        action_probs = self.prob_for_state(s)
        if a > len(action_probs) - 1:
            warnings.warn(
                f"Requested π({a}|{s}), but |A| = {len(action_probs)} - returning 1.0. If {a} is a dummy action you can safely ignore this warning.",
                PaddedMDPWarning,
            )
            return 1.0
        else:
            return action_probs[a]

    def log_prob_for_state_action(self, s, a):
        """Get the log probability for the given state, action

        Args:
            s (int): Current state
            a (int): Chosen action

        Returns:
            (float): Log probability of choosing a from s
        """
        log_action_probs = self.log_prob_for_state(s)
        if a > len(log_action_probs) - 1:
            warnings.warn(
                f"Requested log π({a}|{s}), but |A| = {len(log_action_probs)} - returning 0.0. If {a} is a dummy action you can safely ignore this warning.",
                PaddedMDPWarning,
            )
            return 0.0
        else:
            return log_action_probs[a]

    def path_log_action_probability(self, p):
        """Compute log-likelihood of [(s, a), ..., (s, None)] path under this policy

        N.B. - this does NOT account for the likelihood of starting at state s1 under
            the MDP dynamics, or the MDP dynamics themselves

        Args:
            p (list): List of state-action tuples

        Returns:
            (float): Absolute log-likelihood of the path under this policy
        """

        # We start with probability 1.0
        ll = np.log(1.0)

        # N.b. - final tuple is (s, None), which we skip
        # p = [(s, a), (s, a), ..., (s, None)]
        for s, a in p[:-1]:
            log_action_prob = self.log_prob_for_state_action(s, a)
            if np.isneginf(log_action_prob):
                return -np.inf

            ll += log_action_prob

        return ll

    def get_rollouts(
        self, env, num, max_path_length=None, start_state=None, start_action=None
    ):
        """Sample state-action rollouts from this policy in the provided environment

        Args:
            env (gym.Env): Environment
            num (int): Number of rollouts to sample

            max_path_length (int): Optional maximum path length - episodes will be
                prematurely terminated after this many time steps
            start_state (any): Optional starting state for the policy - Warning: this
                functionality isn't actually supported by the OpenAI Gym class, we just
                hope that the 'env' definition has a writeable '.state' parameter
            start_action (any): Optional starting action for the policy

        Returns:
            (list): List of state-action rollouts
        """
        rollouts = []
        for _ in range(num):
            rollout = []
            s = env.reset()
            if start_state is not None:
                """XXX ajs 28/Oct/2020 The OpenAI Gym interface doesn't actually expose
                a `state` parameter, making it impossible to force a certain starting
                state reliably. Someone needs to publish a standardized MDP interface
                that isn't a steaming pile of b*******.
                """
                assert (
                    "state" in env.__dir__()
                ), "XXX This environment doesn't have a 'state' property - I'm unable to force it into a desired starting state!"
                env.reset()
                env.state = start_state
                s = start_state

            if max_path_length is None or max_path_length > 1:
                for t in it.count():
                    if t == 0 and start_action is not None:
                        a = start_action
                    else:
                        a, _ = self.predict(s)
                    s2, r, done, info = env.step(a)
                    rollout.append((s, a))
                    s = s2

                    path_length = t + 1
                    if done:
                        break
                    if (
                        max_path_length is not None
                        and path_length == max_path_length - 1
                    ):
                        break
            else:
                # Edge case - max path length == 1
                pass

            rollout.append((s, None))
            rollouts.append(rollout)

        return rollouts


class UniformRandomPolicy(Policy):
    """A Uniform Random policy for discrete action spaces"""

    def __init__(self, num_actions):
        """C-tor"""
        self.num_actions = num_actions

    def predict(self, s):
        return np.random.randint(0, self.num_actions), None

    def log_prob_for_state(self, s):
        return np.log(np.ones(self.num_actions) / self.num_actions)


class UniformRandomCtsPolicy(Policy):
    """A Uniform Random policy for continuous action spaces"""

    def __init__(self, action_range):
        """C-tor"""
        self.action_range = action_range
        self.action_delta = self.action_range[1] - self.action_range[0]

    def predict(self, s):
        a = np.array([np.random.rand() * self.action_delta + self.action_range[0]])
        return a, None

    def log_prob_for_state(self, s):
        raise ValueError("Not applicable for continuous action spaces")

    def prob_for_state_action(self, s, a):
        return np.exp(self.log_prob_for_state_action(s, a))

    def log_prob_for_state_action(self, s, a):
        return np.log(1.0 / self.action_delta)


class EpsilonGreedyPolicy(Policy):
    """An Epsilon Greedy Policy wrt. a provided Q function"""

    def __init__(self, q, epsilon=0.1):
        """C-tor

        Args:
            q (numpy array): |S|x|A| Q-matrix
            epsilon (float): Probability of taking a random action. Set to 0 to create
                an optimal stochsatic policy. Specifically,
                    Epsilon == 0.0 will make the policy sample between equally good
                        (== Q value) actions. If a single action has the highest Q
                        value, that action will always be chosen
                    Epsilon > 0.0 will make the policy act in an epsilon greedy
                        fashion - i.e. a random action is chosen with probability
                        epsilon, and an optimal action is chosen with probability
                        (1 - epsilon) + (epsilon / |A|).
        """
        self.q = q
        self.epsilon = epsilon

    def prob_for_state(self, s):
        """Get the action probability vector for the given state

        Args:
            s (int): Current state

        Returns:
            (numpy array): Probability distribution over actions, respecting the
                self.stochastic and self.epsilon parameters
        """

        num_states, num_actions = self.q.shape
        if s > num_states - 1:
            warnings.warn(
                f"Requested π(*|{s}), but |S| = {num_states}, returning [1.0, ...]. If {s} is a dummy state you can safely ignore this warning.",
                PaddedMDPWarning,
            )
            return np.ones(num_actions)

        # Get a list of the optimal actions
        action_values = self.q[s, :]
        best_action_value = np.max(action_values)
        best_action_mask = action_values == best_action_value
        best_actions = np.where(best_action_mask)[0]

        # Prepare action probability vector
        p = np.zeros(self.q.shape[1])

        # All actions share probability epsilon
        p[:] += self.epsilon / self.q.shape[1]

        # Optimal actions share additional probability (1 - epsilon)
        p[best_actions] += (1 - self.epsilon) / len(best_actions)

        return p

    def log_prob_for_state(self, s):
        """Get the action log probability vector for the given state

        Args:
            s (int): Current state

        Returns:
            (numpy array): Log probability distribution over actions
        """
        return np.log(self.prob_for_state(s))


class OptimalPolicy(EpsilonGreedyPolicy):
    """An optimal policy - can be deterministic or stochastic"""

    def __init__(self, q, stochastic=True, q_precision=None):
        """C-tor

        Args:
            q (numpy array): |S|x|A| Q-matrix

            stochastic (bool): If true, this policy will sample amongst optimal actions.
                Otherwise, the first optimal action will always be chosen.
            q_precision (int): Precision level in digits of the q-function. If a
                stochastic optimal policy is requested, Q-values will be rounded to
                this many digits before equality checks. Set to None to disable.
        """
        super().__init__(q, epsilon=0.0)

        self.stochastic = stochastic
        self.q_precision = q_precision

    def prob_for_state(self, s):

        num_states, num_actions = self.q.shape
        if s > num_states - 1:
            warnings.warn(
                f"Requested π(*|{s}), but |S| = {num_states}, returning [1.0, ...]. If {s} is a dummy state you can safely ignore this warning.",
                PaddedMDPWarning,
            )
            return np.ones(num_actions)

        if self.stochastic:

            if self.q_precision is None:
                p = super().prob_for_state(s)
            else:
                # Apply q_precision rounding to the q-function
                action_values = np.array(
                    [round(v, self.q_precision) for v in self.q[s]]
                )
                best_action_value = np.max(action_values)
                best_action_mask = action_values == best_action_value
                best_actions = np.where(best_action_mask)[0]

                p = np.zeros(self.q.shape[1])
                p[best_actions] += 1.0 / len(best_actions)

        if not self.stochastic:
            # Always select the first optimal action
            p = super().prob_for_state(s)
            a_star = np.where(p != 0)[0][0]
            p *= 0
            p[a_star] = 1.0
        return p

    def log_prob_for_state(self, s):
        """Get the action log probability vector for the given state

        Args:
            s (int): Current state

        Returns:
            (numpy array): Log probability distribution over actions
        """
        return np.log(self.prob_for_state())


class BoltzmannExplorationPolicy(Policy):
    """A Boltzmann exploration policy wrt. a provided Q function"""

    def __init__(self, q, scale=1.0):
        """C-tor

        Args:
            q (numpy array): |S|x|A| Q-matrix
            scale (float): Temperature scaling factor on the range [0, inf).
                Actions are chosen proportional to exp(scale * Q(s, a)), so...
                 * Scale > 1.0 will exploit optimal actions more often
                 * Scale == 1.0 samples actions proportional to the exponent of their
                    value
                 * Scale < 1.0 will explore sub-optimal actions more often
                 * Scale == 0.0 will uniformly sample actions
                 * Scale < 0.0 will prefer non-optimal actions
        """
        self.q = q
        self.scale = scale

    def prob_for_state(self, s):
        """Get the action probability vector for the given state

        Args:
            s (int): Current state

        Returns:
            (numpy array): Probability distribution over actions, respecting the
                self.stochastic and self.epsilon parameters
        """
        return np.exp(self.log_prob_for_state(s))

    def log_prob_for_state(self, s):

        num_states, num_actions = self.q.shape
        if s > num_states - 1:
            warnings.warn(
                f"Requested π(*|{s}), but |S| = {num_states}, returning [1.0, ...]. If {s} is a dummy state you can safely ignore this warning.",
                PaddedMDPWarning,
            )
            return np.ones(num_actions)

        log_prob = self.scale * self.q[s]
        total_log_prob = np.log(np.sum(np.exp(log_prob)))
        log_prob -= total_log_prob
        return log_prob


class TorchPolicy(nn.Module, Policy):
    """An abstract base policy class for a single hidden layer MLP that uses PyTorch internally"""

    def __init__(self, in_dim, out_dim, hidden_size, learning_rate=0.01):
        """C-tor

        Args:
            in_dim (int): Input (feature vector) size
            out_dim (int): Output (action vector) size
            hidden_size (int): Size of the hidden layer

            learning_rate (float): Learning rate for optimizer used for training
        """
        super().__init__()
        # super(TorchPolicy, self).__init__()
        # Implementing classes should set these elements
        self.optimizer = None
        self.loss_fn = None
        self.loss_target_type = None
        # raise NotImplementedError

    def forward(self, x):
        """Forward pass through the policy network

        Args:
            x (torch.Tensor): Input feature vector
        """
        raise NotImplementedError

    def predict(self, stoch=True):
        """Predict the next action and state distribution

        N.b. This function matches the API of the stabe-baselines policies.

        Args:
            stoch (bool): If true, sample an action stochastically, otherwise choose
                the most likely action.

        Returns:
            (torch.Tensor): Action vector
        """
        raise NotImplementedError

    def prob_for_state(self, x):
        """Get the probability distribution over actions for the given state

        This method is only applicable for policies with discrete action spaces

        Args:
            x (torch.Tensor): Input feature vector phi(s, a, s')

        Returns:
            (torch.Tensor): Vector of discrete probability distributions over actions
        """
        return torch.exp(self.log_prob_for_state(x))

    def prob_for_state_action(self, x, a):
        """Get the probability for the given state, action

        Args:
            x (torch.Tensor): Input feature vector phi(s, a, s')
            a (torch.Tensor): Chosen action

        Returns:
            (torch.Tensor): Probability of choosing a from phi(s, a, s')
        """
        return torch.exp(self.log_prob_for_state_action(x, a))

    def param_gradient(self):
        """Get the gradient of every parameter in this model as a single vector

        Returns:
            (torch.Tensor): A vector containing the gradient of every parameter in this model
        """
        vec = []
        for param in self.parameters():
            vec.append(param.grad.view(-1))
        return torch.cat(vec)

    def behaviour_clone(
        self, dataset, phi, num_epochs=3000, log_interval=None, weights=None
    ):
        """Behaviour cloning using full-batch gradient descent

        TODO ajs 25/May/2021 Support stochastic gradient descent

        Args:
            dataset (list): List of (s, a) rollouts to clone from
            phi (FeatureFunction): Feature function accepting states and outputting feature vectors

            num_epochs (int): Number of epochs to train for
            log_interval (int): Logging interval, set to 0 to do no logging
            weights (numpy array): Path weights for weighted behaviour cloning

        Returns:
            (float): Final BC loss of the policy
        """

        if weights is None:
            weights = np.ones(len(dataset))

        # Convert states to features, and flatten dataset
        phis = []
        actions = []
        for path, weight in zip(dataset, weights):
            _states, _actions = zip(*path[:-1])
            path_fvs = np.array([phi(s) for s in _states])
            phis.extend(weight * path_fvs)
            actions.extend(_actions)
        phis = torch.tensor(phis)
        actions = torch.tensor(actions, dtype=self.loss_target_type)

        for epoch in range(num_epochs):
            # Run one epoch of training
            self.optimizer.zero_grad()
            loss = self.loss_fn(self(phis), actions)
            loss.backward()
            self.optimizer.step()

            if log_interval is not None and epoch % log_interval == 0:
                print(f"Epoch {epoch}, loss={loss.item()}")

        return loss.detach().numpy()


class MLPGaussianPolicy(TorchPolicy):
    """An MLP-Gaussian policy with a single hidden layer and constant standard deviation

    TODO ajs 13/May/2021 Add support for >1 action
    TODO ajs 13/May/2021 Add support for learned standard deviation
    TODO ajs 25/May/2021 General base class for PyTorch policies

    Supports arbitrary observation spaces, and single-dimensional continuous action spaces.

    Uses PyTorch for MLP implementation and training. Provides a convenience method for
    behaviour cloning.
    """

    def __init__(self, in_dim, out_dim, hidden_size, std=1.0, learning_rate=0.01):
        """C-tor

        # TODO ajs 28/May/2021 Support learned standard deviation

        Args:
            in_dim (int): Input (feature vector) size
            out_dim (int): Output (action vector) size
            hidden_size (int): Size of the hidden layer

            std (float): Fixed standard deviation for this model
            learning_rate (float): Learning rate for optimizer used for training
        """
        super().__init__(in_dim, out_dim, hidden_size, learning_rate)

        if out_dim != 1:
            # TODO ajs 28/May/2021 Support multiple action dimensions
            raise NotImplementedError

        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_dim)
        self.std = std
        self.loss_fn = torch.nn.MSELoss()
        self.loss_target_type = torch.long
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # Input is feature vector phi(s, a, s')
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.float()
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        # Output is mean of a gaussian from which we sample an action
        return x

    def predict(self, x, stoch=True):
        """Predict next action and distribution over states

        N.b. This function matches the API of the stabe-baselines policies.

        Args:
            x (int): Input feature vector

            stoch (bool): If true, sample action stochastically

        Returns:
            (int): Sampled action
            (None): Placeholder to ensure this function matches the stable-baselines
                policy interface. Some policies use this argument to return a prediction
                over future state distributions - we do not.
        """
        mean = self(x)
        if stoch:
            dist = torch.distributions.normal.Normal(mean, self.std)
            a = dist.sample()
        else:
            a = mean
        return a, None

    def log_prob_for_state(self, s):
        """Get the action log probability vector for the given state

        Args:
            s (int): Current state

        Returns:
            (numpy array): Log probability distribution over actions
        """
        raise ValueError(
            "MLPGaussianPolicy supports a single continuous action, this method is only for discrete action spaces"
        )

    def log_prob_for_state_action(self, x, a):
        """Get the log probability for the given state, action

        Args:
            x (int): Current feature vector phi(s, a, s')
            a (int): Chosen action

        Returns:
            (float): Log probability of choosing a from phi(s, a, s')
        """
        mean = self(x)
        dist = torch.distributions.normal.Normal(mean, self.std)
        return dist.log_prob(a)


class MLPCategoricalPolicy(TorchPolicy):
    """An MLP-Categorical policy with a single hidden layer

    Supports arbitrary observation spaces, and discrete action spaces.

    Uses PyTorch for MLP implementation and training. Provides a convenience method for
    behaviour cloning.
    """

    def __init__(self, in_dim, out_dim, hidden_size, learning_rate=0.01):
        """C-tor

        Args:
            in_dim (int): Input (feature vector) size
            out_dim (int): Output (action vector) size
            hidden_size (int): Size of the hidden layer
            learning_rate (float): Learning rate for optimizer used for training
        """
        super().__init__(in_dim, out_dim, hidden_size, learning_rate)

        # self.fc1 = nn.Linear(in_dim, hidden_size)
        self.fc1 = nn.Linear(in_dim, out_dim)
        # self.fc2 = nn.Linear(hidden_size, out_dim)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        self.loss_target_type = torch.long
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # Input is feature vector phi(s, a, s')
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.float()
        x = self.fc1(x)
        # x = nn.functional.relu(x)
        # x = self.fc2(x)
        # Output is vector of categorical log probabilities from which we sample an action
        return x

    def predict(self, x, stoch=True):
        """Predict next action and distribution over states

        N.b. This function matches the API of the stabe-baselines policies.

        Args:
            x (int): Input feature vector

            stoch (bool): If true, sample action stochastically

        Returns:
            (int): Sampled action
            (None): Placeholder to ensure this function matches the stable-baselines
                policy interface. Some policies use this argument to return a prediction
                over future state distributions - we do not.
        """
        probs = torch.exp(self(x))
        if stoch:
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
        else:
            a = torch.argmax(probs)
        return a, None

    def log_prob_for_state(self, x):
        """Get the action log probability vector for the given feature vector

        Args:
            x (numpy array): Current feature vector

        Returns:
            (numpy array): Log probability distribution over actions
        """
        return self(x)

    def log_prob_for_state_action(self, x, a):
        """Get the log probability for the given state, action

        Args:
            x (int): Current feature vector phi(s, a, s')
            a (int): Chosen action

        Returns:
            (float): Log probability of choosing a from phi(s, a, s')
        """
        return self(x)[int(a.item())]
