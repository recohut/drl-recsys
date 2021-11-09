from rl import *
from recsim.agents import  full_slate_q_agent, random_agent
from irl_agent import InverseRLAgent
import tensorflow.compat.v1 as tf
from numpy import load
import pandas as pd
import matplotlib.pyplot as plt
import sys
import calendar;
import time;

def evalRun_one_episode(env, agent, agent_name="random", max_steps_per_episode=100):
    """
        Runs one episode with the given configuration 

        Args:
            env: recommendation gym environnement
            agent: recommendation agent (Full Slate-Q-learning || Rnadom/naive agent)
            agent_name: name of the chosen agent ('random' || 'fsq')

        Returns:

            step_number: length of the episode
            total_reward: total watching_time and quality for this episode
            time_dif: execution time of this episode
            total_qual/step_number: Average total quality of the episode
            q_videos: list of videos (clicked throughout the episdoe) qualities 
            w_videos: list of videos (clicked throughout the episdoe) watching time

        """
    observation = env.reset()
    action = agent.begin_episode(observation)
    step_number = 0
    total_watch = 0.
    q_videos = []
    w_videos = []
    total_qual = 0
    start_time = time.time()
    total_length_videos = 0
    while True:
        observation, reward, done, info, _ = env.step(action)

        for j in range(len(observation['response'])):
            if observation['response'][j]['click'] == 1:
                index = action[j]
                total_length_videos += list(observation['doc'].values())[index][-1]
                total_watch += reward[1]
                total_qual += reward[0]
                q_videos += [reward[0]]
                w_videos += [reward[1]]

        # Update environment-specific metrics with responses to the slate.
        env.update_metrics(observation['response'], info)
        step_number += 1

        if done:
            break
        elif step_number == max_steps_per_episode:
            # Stop the run loop once we reach the true end of episode.
            break
        else:
            if agent_name == "random":
                action = agent.step(observation)
            elif agent_name == "fsq":
                action = agent.step(reward[1], observation)
            else:
                print("agent name is not correct, please select (random || fsq))")

    agent.end_episode(reward[1], observation)
    time_diff = time.time() - start_time

    return step_number, total_watch, time_diff, total_qual/step_number, q_videos, w_videos

def clicked_quality_reward(responses):
    """Calculates the total clicked watchtime from a list of responses.

    Args:
      responses: A list of IEvResponse objects

    Returns:
      reward: A float representing the total watch time from the responses
      """
    qual = 0.0
    watch = 0.0
    for response in responses:
        if response.clicked:
            qual += float(response.quality)
            watch += float(response.watch_time)
    return [qual, watch]


def create_agent_random(slate_size, num_candidates, random_seed=0):
    action_space = spaces.MultiDiscrete(num_candidates * np.ones((slate_size,)))
    return random_agent.RandomAgent(action_space, random_seed)

# define reward functions
def clicked_engagement_reward(responses):
    reward = 0.0
    for response in responses:
        if response.clicked:
            reward += response.watch_time
    return reward


            
def main():

    # read configuration inputs
    max_episode = int(sys.argv[1])
    slate_size = int(sys.argv[2])
    num_candidates = int(sys.argv[3])
    file_states = sys.argv[4]
    file_policy = sys.argv[5]
    # Initialisation...
    results_r = []
    results_f = []
    results = []

    env_config = {
        'num_candidates': num_candidates,
        'slate_size': slate_size,
        'resample_documents': True,
        'seed': 0,
        'reward_function':clicked_engagement_reward
    }

    # User simulation environment: interest evolution model presented in the paper of SlateQ
    recsim_gym_env = interest_evolution.create_environment(env_config)

    # Load the learned policy from the Expert IRL model as well as the resulted/associated states
    states = load('./datasets_states/'+file_states, allow_pickle=True)
    policy_ = load('./datasets_states/'+file_policy, allow_pickle=True)

    # Instanciate the Expert IRL agent 
    agent_irl = InverseRLAgent(recsim_gym_env, states, policy_, num_cand=num_candidates,
              slate_size=slate_size, max_steps_per_episode=100, ml_model=False)

    for i in range(max_episode):
        steps, reward, time_, found, exp_watch, user_metrics = agent_irl.run_one_episode()
        results += [[i,steps, reward, time_, found, exp_watch, user_metrics]]

        user_reward = []
        episode_steps = []
        episode_total_quality = []
        episode_videos_length = []


    for i in range(len(results)):
        episode_steps += [results[i][1]]
        user_reward += [results[i][2]]
        episode_videos_length += [results[i][6][1]]
        episode_total_quality += [results[i][6][3]]


    # configuration for running  recNaive
    env_config1 = {
    'num_candidates': num_candidates,
    'slate_size': slate_size,
    'resample_documents': True,
    'seed': 0,
    'reward_function':clicked_quality_reward
    }

    recsim_gym_env = interest_evolution.create_environment(env_config1)
    agent = create_agent_random(slate_size, num_candidates)

    for i in range(max_episode):
        steps_r, watch, time_r, q, q_vid, w_vid = evalRun_one_episode(recsim_gym_env, agent, "fsq")
        results_r += [[i,steps_r, watch, time_r, q, q_vid, w_vid ]]
        
    episode_steps_r = []
    episode_ratio_watch_r = []
    episode_total_quality_r = []
    episodes_qv = []
    episodes_wv = []

    for i in range(len(results_r)):
        episode_steps_r += [results_r[i][1]]
        episode_ratio_watch_r += [results_r[i][2]]
        episode_total_quality_r += [results_r[i][4]]
        episodes_qv += [results_r[i][5]]
        episodes_wv += [results_r[i][6]]


    # configuration for running recFSQ
    for j in range(8):
        agent = full_slate_q_agent.FullSlateQAgent(tf.Session(config=tf.ConfigProto(allow_soft_placement=True)),
        recsim_gym_env.observation_space, recsim_gym_env.action_space)

    for i in range(250):
        steps_f, watch, time_f, q, q_vid, w_vid = evalRun_one_episode(recsim_gym_env, agent, "fsq")
        results_f += [[i,steps_f, watch, time_f, q, q_vid, w_vid]]
        print("episode ",i*j)


    episode_steps_f = []
    episode_ratio_watch_f = []
    episode_total_quality_f = []
    episodes_qvf = []
    episodes_wvf = []


    for i in range(len(results_f)):
        episode_steps_f += [results_f[i][1]]
        episode_ratio_watch_f += [results_f[i][2]]
        episode_total_quality_f += [results_f[i][4]]
        episodes_qvf += [results_f[i][5]]
        episodes_wvf += [results_f[i][6]]


    # save histogram result into image
    data_quality = [sum(episode_total_quality_f) / max_episode, sum(episode_total_quality_r) / max_episode, sum(episode_total_quality) / max_episode]
    data_watch_time = [sum(episode_ratio_watch_f), sum(episode_ratio_watch_r), float(sum(user_reward))]
    data = np.array([data_quality, data_watch_time])
    pd.DataFrame(data, columns=["Full_Q", "Naive_random_", "FEBR"], index=["#Quality","#Total_Watch_time"])

    approaches = ["RecFSQ","RecNaive", "RecFEBR"]
    ts = calendar.timegm(time.gmtime())
    build_histo_q(approaches, data, ts)
    build_histo_w(approaches, data, ts)



def build_histo_q(approaches_list, data, time_stamp):
    plt.bar(approaches, data[0])
    plt.ylabel('Average total quality')
    plt.yticks(np.arange(-0.2, 0.5, 0.05))
    plt.savefig("./eval_results/q_t_compar"+str(time_stamp)+".png", bbox_inches='tight')
    print("quality result is saved into eval_result folder")


def build_histo_w(approaches_list, data, time_stamp):
    plt.bar(approaches, data[1])
    plt.ylabel('Total watch time (s)')
    plt.savefig("./eval_results/w_t_compar"+str(time_stamp)+".png", bbox_inches='tight')
    print("watching_time result is saved into eval_result folder")

if __name__ == '__main__':
    main()