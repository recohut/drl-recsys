from utils import proba_trans, generatesIndexSlates
from recsim import choice_model
from ExpertRecEval import *




def policy_iteration(list_states, rewards, _policy, gamma, num_candidate, size_slate, max_iteration=1000, eps=0.1):

    """
    Implementation of a static policy iteration function for policy optimisation 

  inputs:
    list_states : list of experts' states generated from  trajectories
    rewards     : reward matrice for all the states
    _policy     : policy matrice (initialized by the MaxEnt-IRL algorithm)
    gamma       : float - RL discount factor
    num_candidate: The size of the corpus video candidate.
    size_slate: The number of recommended videos inside the proposed list.
    max_iter  : maximum number of iteration before stop
    eps       : float - threshold for convergence condition
   


  returns:
    values    vector of  estimated values
    new_policy    policy stochastic_matrice
  """


    v = np.zeros([len(list_states)])
    action_list = generatesIndexSlates(num_candidate, size_slate)
    old_policy = _policy
    new_policy = np.zeros([len(list_states)], dtype=int)

    for i in range(max_iteration):
        print("iteration_policy: ", i)
        while(True):
            iteration_count = 0
            prev_v = np.copy(v)
            for s in range(len(list_states)):
                v[s] = sum([proba_trans(list_states[s1], list_states[s], action_list[int(old_policy[s])]) * (rewards[s] + gamma * prev_v[s1])
                                for s1 in range(len(list_states))])
            m = max([abs(v[s] - prev_v[s]) for s in range(len(list_states))])
            if m <= eps:
                print('Value converged at iteration# %d.' % (iteration_count + 1))
                break

        for s in range(len(list_states)):
            q_sa = np.zeros([len(action_list)])
            for a in range(len(action_list)):
                q_sa[a] = sum([proba_trans(list_states[s_], list_states[s], action_list[a]) * (rewards[s] + gamma * v[s_])
                               for s_ in range(len(list_states))])
            new_policy[s] = np.argmax(q_sa)
    
        if np.all(old_policy == new_policy):
            print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        old_policy = new_policy

    return v, new_policy



def value_iteration(list_states, rewards, gamma, num_candidate, size_slate, max_iter=1000, error=0.01):

    """
    Implementation of a static value iteration function for policy optimisation 

  inputs:
    list_states : list of experts' states generated from  trajectories
    rewards     : reward matrice for all the states
    gamma       : float - RL discount factor
    num_candidate: The size of the corpus video candidate.
    size_slate: The number of recommended videos inside the proposed list.
    max_iter  : maximum number of iteration before stop
    error       : float - threshold for convergence condition
   


  returns:
    values    vector of  estimated values
    policy    policy stochastic_matrice
  """

    values = np.zeros([len(list_states)])
    list_action = generatesIndexSlates(num_candidate, size_slate)
    # estimate values
    for j in range(max_iter):
        iteration_count = 0
        values_tmp = values.copy()

        for s in range(len(list_states)):
            values[s] = max(
                [sum([proba_trans(list_states[s1], list_states[s], a) * (rewards[s] + gamma * values_tmp[s1])
                      for s1 in range(len(list_states))]) / len(list_states)
                 for a in list_action])

        m = max([abs(values[s] - values_tmp[s]) for s in range(len(list_states))])
        print("max_nn= ", m)

        if m < error:
            print('Value-iteration converged at iteration# %d.' % (iteration_count + 1))
            break
        print('Value-iteration  at iteration# %d.' % iteration_count)
        iteration_count += 1

    policy = np.zeros([len(list_states), len(list_action)])
    for s in range(len(list_states)):
        v_s = np.array(
            [sum([proba_trans(list_states[s1], list_states[s], a) * (rewards[s] + gamma * values[s1])
                  for s1 in range(len(list_states))])
             for a in list_action])
        policy[s, :] = np.transpose(v_s / np.sum(v_s))

    return values, policy



