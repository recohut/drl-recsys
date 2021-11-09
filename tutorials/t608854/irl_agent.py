
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import pickle

from utils import *


class InverseRLAgent():
    """A  recommender system agent that implements the policy learned by
      Maximum Entropy inverse reinforcement learning model."""

    def __init__(self, env, states_list, policy, num_cand, slate_size, max_steps_per_episode=100, 
    	        ml_model=False,filename='expert_bestModel.sav'):
        
    	"""
    	Args:
    		env: The RecSim Gym user environment, e.g. interest evolution environment
    		states_list: Dataset of expert states
    		policy: The learned expert policy produced by MaxEnt-IRL model
    		num_cand: The size of the corpus video candidate
    		slate_size: The number of recommended videos 
    		max_steps_per_episode: The length of a user session (size of an episode)
    		ml_model: Should be set to True in the case you want to use a classification 
    				  model (learned on the dataset) for prediction. Otherwise, you use the 
    				  comparaison model with similarities margins
    		filename: If ml_model=True, you have to specify your serialized machine learning model

    	"""
        self.env = env
        self.states = states_list
        self.policy = policy
        self.list_actions = generatesIndexSlates(num_cand, slate_size)
        self.max_steps_per_episode = max_steps_per_episode
        # number of user states that matched a given expert state
        self.count = 0
        # total watch time by the expert during a user episode
        self.expert_watch = 0
        # total quality delivered by following the expert policy
        self.deleg_quality = 0

        self.deleg_q_temp = 0

        self.ml_model = ml_model

        self.loaded_model = None
        # list to save all state's qualities through a given session (episode)
        self.list_quality_videos = []
        # list to save watching time of selected videos through a given session (episode)
        self.list_watching_time = []

    def step(self, observation):

    	"""Receives observations of environment and returns a slate.

	    Args:
	      observation: A dictionary that stores all the observations including:
	        - user: A list of floats representing the user's observed state
	        - doc: A list of observations of document features
	        - response: A vector valued response signal that represent user's
	          response to each document

	    Returns:
	      slate: An integer array of size _slate_size, where each element is an
	        index in the list of document observvations.
	    """


        if not self.ml_model:
            for i in range(len(self.states)):
                if self.find_state_(observation, self.states[i], margin_score=0.5, margin_interests=0.5):
                    #print("state found ")
                    for j in self.states[i][1]:
                        if j[0] == 1:
                            self.expert_watch += j[5]
                            self.deleg_q_temp = (j[1] + j[2] + j[3] + j[4]) / 4
                            self.deleg_quality += self.deleg_q_temp

                    self.count += 1
                    return self.list_actions[np.argmax(self.policy[i])], 1
            #print("state not found")
            return self.list_actions[np.random.randint(0, len(self.list_actions), dtype=int)], 0
        else:
            self.loaded_model = pickle.load(open(filename, 'rb'))
            s = self.loaded_model.predict([observation['user']])

            return self.list_actions[int(s)], 1



    def find_state(self, user_state, expert_state, margin_features=0.1):
        for j in range(0, len(user_state)):
            if abs(user_state['user'][j] - expert_state[0][j + 1]) > margin_features:
                return False
        return True

    def find_state_(self, user_state, expert_state, margin_score=0.1, margin_interests=0.5):

    	"""
    	Implements a simple classification algorithm to compare states accordings to some margins.
    	This function could be overwritten or modified depending on what criteria or similarities 
    	you want to evaluate.
    	Args:
    		user_state:  A list of floats representing the user's observed state from the user observation
    		expert_satte: A list of floats representing the expert's observed state from the  dataset
    		margin_score: to compare video's quality
    		margin_interests: to compare user and expert's interests.

    	Returns:
    		True if the state has been found, and False if it is not;


    	"""

        assert (len(expert_state[0][1:]) == len(user_state['user'])
                ), 'user interests size does not match'

        for i in range(len(list(user_state['doc'].values()))):
            if not (list(user_state['doc'].values())[i][:len(expert_state[0][1:])] == list(expert_state[2].values())[i][
                                                                        :len(expert_state[0][1:])]).all():
                #print("false1 ")
                return False
        if (abs(user_state['user'][1] - expert_state[0][2]) > margin_interests):
            #print("false2 ",user_state['user'], expert_state[0])
            return False
        for i in range(len(list(user_state['doc'].values()))):
            a = np.dot(user_state['user'], list(user_state['doc'].values())[i][:len(expert_state[0][1:])]) * \
                list(user_state['doc'].values())[i][-2]
            b = np.dot(expert_state[0][1:], list(expert_state[2].values())[i][:len(expert_state[0][1:])]) * \
                (sum(list(expert_state[2].values())[i][len(expert_state[0][1:]):]) /
                 len(list(expert_state[2].values())[i][len(expert_state[0][1:]):]))
            if abs(a - b) > margin_score:
                #print("false3 ", abs(a - b))
                return False
        return True

    def run_one_episode(self):

    	"""
		Runs one episode with the given configuration 

		Returns:

		step_number: length of the episode
		total_reward: total watching_time and quality for this episode
		time_dif: execution time of this episode
		c_found: number of user states that have been matched to expert states by the classifier
		expert_watch: total watching time by state experts for user states that are similar to those experts states
		total_clicked: number of clicked videos throughout this episode
		total_length_videos: total duration time of the clicked videos
		total_deleg_q = total quality calculated by slates recommended by the expert policy
		total_episode_q = total quality of the episode for all the clicked watched videos 

    	"""
    	# Initialize the envronment
        observation = self.env.reset()
        action, test = self.step(observation)
        step_number = 1
        total_reward = 0.
        self.count = 0
        self.expert_watch = 0
        self.deleg_quality = 0
        total_quality_exp = 0
        total_quality_not_found = 0
        total_clicked = 0
        total_length_videos = 0
        start_time = time.time()
        c_found = 0
        while True:
        	# execute the action and receives the reward and the new observation from the environment
            observation, reward, done, info, _ = self.env.step(action)
            for j in range(len(observation['response'])):
            	# if the user has clicked on the video
                if observation['response'][j]['click'] == 1:
                	# user state is found ( matched to a similar expert state)
                    if test == 1:
                        c_found += 1
                        if reward != None:
                            self.list_watching_time += [reward]
                            self.list_quality_videos += [self.deleg_q_temp]
                    elif test == 0:
                        total_quality_not_found += float(observation['response'][j]['quality'])
                        if reward != None:
                            self.list_quality_videos += [float(observation['response'][j]['quality'])]
                            self.list_watching_time += [reward]
                    index = action[j]
                    total_length_videos += list(observation['doc'].values())[index][-1]

                    total_clicked += 1

                    total_reward += reward

                    break
            self.env.update_metrics(observation['response'], info)

            step_number += 1

            if done:
                break
            elif step_number == self.max_steps_per_episode:
                # Stop the run loop once we reach the true end of episode.
                break
            else:
            	# receive the new slate from the agent (classifier)
                action, test = self.step(observation)

        time_diff = time.time() - start_time
        total_delga_q = self.deleg_quality/c_found
        total_episode_q = (total_quality_not_found + self.deleg_quality)/total_clicked

        return step_number, total_reward, time_diff, c_found, self.expert_watch,\
               [total_clicked, total_length_videos, total_del_q, total_episode_q]
        

    def videos_info(self):
    	""" Returns the lists of watching time and the delivered 
    		quality for states of the associated episode"""
        return self.list_watching_time, self.list_quality_videos

def clicked_engagement_reward(responses):
    reward = 0.0
    for response in responses:
        if response.clicked:
            reward += response.watch_time
    return reward

