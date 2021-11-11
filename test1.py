from recsim.agents import full_slate_q_agent, random_agent
from rl import *
from utils import *
from recsim.agents import  full_slate_q_agent, random_agent, cluster_bandit_agent, greedy_pctr_agent
from irl_agent import InverseRLAgent
import tensorflow.compat.v1 as tf
import time
from recsim.environments import interest_evolution
from recsim.simulator import runner_lib


def create_agent_full_slate(sess, environment, eval_mode, summary_writer=None):
    kwargs = {
        'observation_space': environment.observation_space,
        'action_space': environment.action_space,
        'summary_writer': summary_writer,
        'eval_mode': eval_mode,
    }
    return full_slate_q_agent.FullSlateQAgent(sess, **kwargs)


def create_agent_random(slate_size, random_seed=0):
    action_space = spaces.MultiDiscrete(num_candidates * np.ones((slate_size,)))
    return random_agent.RandomAgent(action_space, random_seed)


def clicked_evaluation_reward(responses):
    reward = 0.0
    for response in responses:
        if response.clicked:
            if response.evaluated:
                reward += 1
    return reward

def evalRun_one_episode(env, agent, agent_name="random", max_steps_per_episode=100):
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

def clicked_engagement_reward(responses):
    reward = 0.0
    for response in responses:
        if response.clicked:
            reward += response.watch_time
    return reward


if __name__ == '__main__':
    from recsim.environments import interest_evolution
    slate_size = 2
    num_candidates = 5
    env_config = {
        'num_candidates': num_candidates,
        'slate_size': slate_size,
        'resample_documents': True,
        'seed': 0,
    }


    env_config1 = {
        'num_candidates': num_candidates,
        'slate_size': slate_size,
        'resample_documents': True,
        'seed': 0,
        'reward_function': clicked_quality_reward
    }
    # User simulation environment: interest evolution model presented in the paper of SlateQ
    recsim_gym_env = interest_evolution.create_environment(env_config1)

    results_f = []

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        #agent = full_slate_q_agent.FullSlateQAgent(sess,
                                                   #recsim_gym_env.observation_space, recsim_gym_env.action_space)


        agent = greedy_pctr_agent.GreedyPCTRAgent(sess,recsim_gym_env.observation_space,recsim_gym_env.action_space)
        #agent = cluster_bandit_agent.ClusterBanditAgent(sess,recsim_gym_env.observation_space,recsim_gym_env.action_space)


        for i in range(10):

            steps_f, watch, time_f, q, q_vid, w_vid = evalRun_one_episode(recsim_gym_env, agent, "fsq")
            results_f += [[i, steps_f, watch, time_f, q, q_vid, w_vid]]
            print("episode ", i)
            sess.run(tf.global_variables_initializer())

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


    print(episode_steps_f)
    print(episode_ratio_watch_f)
    print(episode_total_quality_f)
    print(episodes_qvf)
    print(episodes_wvf)