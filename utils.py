import math
import itertools
import numpy as np



def normalize(vals):
    """
    normalize to (0, max_val)
    input:
      vals: 1d array
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)

def softmax(vector):
    """Computes the softmax of a vector."""
    normalized_vector = np.array(vector) - np.max(vector)  # For numerical stability
    return np.exp(normalized_vector) / np.sum(np.exp(normalized_vector))



def nCr(n, r):

    """ compute the combination nCr"""
    f = math.factorial
    return f(n) / f(r) / f(n - r)

def generatesIndexSlates(num_candidates, slate_size):
    """ generate a random list of size slate_size and elements 
        in the interval [0,num_candidate] """

    stuff = np.linspace(0, num_candidates - 1, num_candidates, dtype=int)
    list = []
    for L in range(0, len(stuff) + 1):
        for subset in itertools.combinations(stuff, L):
            if len(subset) == slate_size:
                check = [(subset[i] + 1 == subset[i+1]) for i in range(len(subset) - 1)]
                if not check.__contains__(False):
                    list.append(subset)

    return list

def compare_state(obs1, obs2, margin_noise=0.3, margin_features=0.1):
    """

    Compare 2 user observations based on the configured margins

    :param obs1: user's state => observation 1
    :param obs2: user's state => observation 2
    :param margin_noise: acceptable difference between noisy states of obs1 and obs2
    :param margin_features: acceptable difference between each features(user interests) of obs1 and obs2
    :return: true if obs1 is similar to obs2, false if otherwise


    """
    if abs(obs1[0] - obs2[0]) > margin_noise:
        print("False")
        return False
    for i in range(1, len(obs1)):
        if abs(obs1[i] - obs2[i]) > margin_features:
            print("False")
            return False
    print("True")
    return True

def compare_state_(state1, state2, margin_score=0.1, margin_interests=0.1):

    """
    Compare 2 experts states from the constructed dataset for size reduction (by removing similar states)

    """

    for i in range(len(list(state1[2].values()))):
        if not (list(state1[2].values())[i][:len(state1[0][1:])] == list(state2[2].values())[i][:len(state1[0][1:])]).all():
            return False
    if abs(sum(state1[0][1:]) * state1[0][0] - sum(state2[0][1:] * state2[0][0])) > margin_interests:
        return False
    for i in range(len(list(state1[2].values()))):
        a = score_document(state1[0][1:], list(state1[2].values())[i])
        b = score_document(state2[0][1:], list(state2[2].values())[i])
        if abs(a - b) > margin_score:
            return False
    return True



def proba_trans(s_next, s_current, action_slate):
    """

    Calculate the transition probability from state s_current 
    to state s_next by taking the action action_slate

    """
    docs = s_current[2]
    scores = score_slate(s_current, action_slate)
    #print("slate's score= ", scores)
    prob = 0
    i = 0
    for index in action_slate:
        prob += scores[i] * transition_choice(s_current, s_next, list(docs.values())[index])
        i += 1
    return prob


def score_document(user_interests, doc_obs):
    '''
    Compute the importance (reflects attraction and quality and engagement effect ) of a video to a given user

    :param user_interests: user interest list
    :param doc_obs: doc features list
    :return: score = interest_for_video_features * video_quality in [0,1] 
    '''
    s = np.dot(user_interests, doc_obs[:len(user_interests)])
    q = sum(doc_obs[len(user_interests):]) / len(doc_obs[len(user_interests):])
    return s * q

def score_slate(state, slate):
    """
    Score the documents of a recommended slate in a given state

    :param state: user or expert state/observation
    :param slate: list of index of the chosen videos
    :return: scored value calculated by the sigmoid function f=1/(1+exp(-x))

    """
    doc_obs = state[2]
    scores = np.array([])
    for i in slate:
        scores = np.append(scores, score_document(state[0][1:], list(doc_obs.values())[i]))
    return 1 / (1 + np.exp(- scores))

def transition_choice(c_start, s_dest, doc):
    '''
    calculate P(s'/s,A,i) = P(s'/s,A',i) = P(s'/s,i)

    :param c_start: s
    :param s_dest: s'
    :param doc: i
    :return: probability of transition from s to s'
    '''

    num_feat = len(c_start[0][1:])
    score_start = score_document(c_start[0][1:], doc)
    score_dest = score_document(s_dest[0][1:], doc)
    result = score_dest - score_start
    return 1 / (1 + np.exp(-result))


def find_obs(obs, states_list):

    """
    linear search of a state within a states list
    N.B: It is an example to serve tests, it is not optimal for large list !

    """
    for i in range(len(states_list)):
        if (obs == states_list[i][0]).all():
            return i
    return -1

def generate_random_policy(states, action=None, num_candidates=10, slate_size=3):

    """

    A proposed methode to randomly initialize a policy for the policy iteration algorithm
    in this example, our policy is a uniform probability of selecting a list of size slate_size

    """
    policy_mat = np.zeros((len(states)), dtype=int)
    n = nCr(num_candidates, slate_size)
    for i in range(len(states)):
        policy_mat[i] = np.random.randint(0, n, dtype=int) / n

    return policy_mat
