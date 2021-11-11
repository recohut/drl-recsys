'''
Implementation of MaxEnt-IRL model for FBER recommendation system, based on 
the approach of Ziebart et al. 2008 paper: Maximum Entropy Inverse Reinforcement Learning

'''

from collections import *
from rl import *
from utils import *
from recsim.agents import full_slate_q_agent, random_agent

MAX_STATES = 1000

def generate_trajectories(env, n_paths, n_steps=100, num_candidates=5, slate_size=2):
    """
        Generate the experts episodes through the expert simulation environment
        Args:
            n_paths : number of trajectories (episodes)
            n_steps: maximum length of a trajectory
            num_candidate: size of the corpus of each state
            slate_size: the size of a recommendation slate

        Returns:
        trajectories: list of expert episodes (expert_obs,video_obs,response) 
        states: list of all observations (states) constructing the trajectories
    """
    trajectories = []
    states = []
    Observations = namedtuple('Observation', ['user_state', 'responses', 'documents', 'done'])
    expert_gym_env = recsim_gym.RecSimGymEnv(env)
    done = False
    print(n_steps)
    # Create agent
    action_space = spaces.MultiDiscrete(num_candidates * np.ones((slate_size,)))
    agent = random_agent.RandomAgent(action_space, random_seed=0)
    for i in range(0, n_paths):
        observation_0 = expert_gym_env.reset()
        #print(observation_0['doc'])
        recommendation_slate_0 = agent.step(0,observation_0)
        path = []
        count = 1
        while not done and count <= n_steps:
            observation_next, _, done, _, _ = expert_gym_env.step(recommendation_slate_0)
            
            path.append(Observations(observation_next['user'],
                                     observation_next['response'], observation_next['doc'], done))
            count += 1
            states.append(Observations(observation_next['user'],
                                       observation_next['response'], observation_next['doc'], done))
            recommendation_slate_0 = agent.step(observation_next)

        trajectories.append(path)

    return trajectories, states



def computeSVF(trajs, states_list, policy, rl_algo, n_steps=100,
               margin_score=0.2, margin_interests=0.1, num_candidates=10, slate_size=3):

    """
        Compute observation visitation  frequency p(s| theta) using dynamic programming
        Args:
            trajs: list of expert trajectories
            states_list: list of experts' states produced from  trajectories
            policy: the xpert policy learned by maxEnt-IRL model
            rl_algo: specifying which RL to optimize the policy and rewards
            n_steps: maximal length of a trajectory
            margin_score: to compare video's quality
            margin_interests: to compare user and expert's interests. 
            num_cand: The size of the corpus video candidate.
            slate_size: The number of recommended videos inside the proposed list.

        Returns:
        p: probability matrice of visiting states
        states_list: final expert states list after deleting similar states based on the comparison model (used for test purposes)
    """

    states_matrices = np.zeros((len(trajs[0]), len(trajs[0])), float)
    action_list = generatesIndexSlates(num_candidates, slate_size)
    print("start n_states= ", len(states_list))
    np.fill_diagonal(states_matrices, 1.0)

    if len(trajs[0]) < n_steps:
        mat = np.zeros((len(trajs[0]), (n_steps - len(trajs[0]))))
        print(states_matrices.shape, ",", n_steps, ",", mat.shape)
        states_matrices = np.append(states_matrices, mat, 0)


    for i in range(1, len(trajs)):
        states_ = np.zeros((len(trajs[i]), len(trajs[i])), float)
        np.fill_diagonal(states_, 1)
        if len(trajs[i]) < n_steps:
            mat = np.zeros((len(trajs[i]), (n_steps - len(trajs[i]))))
            states_ = np.append(states_, mat, 0)
        states_matrices = np.append(states_matrices, states_, 0)

 
    assert (states_matrices.shape[0] == len(states_matrices))
    states_matrices_temp = states_matrices.copy()
    states_list_temp = states_list.copy()
    for i in range(len(states_matrices)):
        for j in range(len(states_matrices)):
            # print(states_matrices.shape,',',len(states_list))
            if states_matrices[j][0] == 1 and i != j:
                if compare_state_(states_list[i], states_list[j], margin_score, margin_interests):
                    states_matrices_temp[i][0] += 1


    states_list = states_list_temp
    states_matrices = states_matrices_temp
    assert (states_matrices.shape[0] == len(states_list))
    states_matrices[:, 0] = states_matrices[:, 0] / len(trajs)

    for s in range(len(states_list)):
        for t in range(n_steps - 1):
            if rl_algo == "value_iter":
                states_matrices[s, t + 1] = sum(
                    [sum([states_matrices[i, t] * proba_trans(states_list[s], states_list[i], a1)
                          * policy[i, action_list.index(a1)]
                          for a1 in action_list]) for i in range(len(states_list))])
            else:
                states_matrices[s, t + 1] = sum(
                    [sum([states_matrices[i, t] * proba_trans(states_list[s], states_list[i], action_list[policy[i]])])
                     for i in range(len(states_list))])

            states_matrices[s, t + 1] = states_matrices[s, t + 1] / (len(states_list) * len(action_list))
        print("obs", s, " finished")

    p = np.sum(states_matrices, 1)
    print("fin n_states= ", len(states_list_temp))

    return p, states_list



def maxEnt_irl(list_states, gamma, trajectories, rl_algo,
               lr=0.1, n_iters=10000, candidate=10, slate=3, n_steps=10):

    """
  Maximum Entropy Inverse Reinforcement Learning for expert recommendations

  inputs:
    states_list : list of experts' states generated from  trajectories
    gamma       : float - RL discount factor
    trajectories: list of expert trajectories (demonstrations)
    rl_algo     : choose the RL algorithm to be used for policy optimisation: value_iter || policy_iter     
    lr          :float - learning rate
    n_iters     :int - number of optimization steps

  returns
    rewards     : vector of recoverred state rewards
    policy      : stochastic policy associated with the optimal rewards, it contains the probability P(s'|s) 
                  of visiting a state s' starting from state s
  """

    feat_map =[]
    for i in range(len(list_states)):
        temp = np.array([])
        temp = np.append(temp, list_states[i][0])
        for j in range(len(list_states[i][1])):
            if list_states[i][1][j][0] == 1:
                temp = np.append(temp, list_states[i][1][j][2:])
                break
        feat_map.append(temp)
    feat_map = np.array(feat_map)
    print(feat_map)
    # init weights
    theta = np.random.uniform(size=(feat_map.shape[1],))
    print("feat_map ", feat_map.shape)
    # calculation of feature expectations
    feat_exp = np.zeros([feat_map.shape[1]])
    feat_exp = sum(feat_exp) / len(trajectories)
    policy_mat = generate_random_policy(states, num_candidates=candidate, slate_size=slate)
    # training
    for iteration in range(n_iters):
        print("iteration= ", iteration)
        if iteration % (n_iters / 20) == 0:
            print('iteration: {}/{}'.format(iteration, n_iters))

        # compute reward function
        rewards = np.dot(feat_map, theta)

        if rl_algo == "value_iter":
            value, policy_ = value_iteration(list_states, rewards, gamma, candidate, slate)
            svf = computeSVF(trajectories, list_states, policy_, "value_iter", n_steps, num_candidates=candidate, slate_size=slate_size)

        elif rl_algo == "policy_iter":
            value, policy_ = policy_iteration(list_states, rewards, policy_mat, gamma, candidate, slate)
            print("----->", policy_)
            svf = computeSVF(trajectories, list_states, policy_, "policy_iter", n_steps)
            policy_mat = policy_
        else:
            print("algo name doesn't exist")
            break

        grad = feat_exp - feat_map.T.dot(svf[0])
        theta += lr * grad

    rewards = np.dot(feat_map, theta)
    return normalize(rewards), policy_


