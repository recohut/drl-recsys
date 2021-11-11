from rl import *
from utils import *
import random
from recsim.agents import  full_slate_q_agent
from irl_agent import InverseRLAgent
import tensorflow.compat.v1 as tf
import time
from numpy import load


def run_one_episode(env, agent, max_steps_per_episode=100):
    observation = env.reset()
    action = agent.begin_episode(observation)
    step_number = 0
    total_watch = 0.
    total_qual = 0
    start_time = time.time()
    total_length_videos = 0
    while True:
        observation, reward, done, info, _ = env.step(action)

        for j in range(len(observation['response'])):
            if observation['response'][j]['click'] == 1:
                index = action[j]
                total_length_videos += list(observation['doc'].values())[index][-1]
                break

        # Update environment-specific metrics with responses to the slate.
        env.update_metrics(observation['response'], info)

        total_watch += reward[1]
        total_qual += reward[0]
        step_number += 1

        if done:
            break
        elif step_number == max_steps_per_episode:
            # Stop the run loop once we reach the true end of episode.
            break
        else:
            action = agent.step(reward[1], observation)

    agent.end_episode(reward[1], observation)
    time_diff = time.time() - start_time
    print("hhhhhhhhhhh",total_length_videos,"fffff",total_watch)

    return step_number, total_watch/total_length_videos, time_diff, total_qual

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

if __name__ == '__main__':
    def clicked_evaluation_reward(responses):
        reward = 0.0
        for response in responses:
            if response.clicked:
                if response.evaluated:
                    reward += 1
        return reward


    slate_size = 2
    num_candidates = 5
    env_config = {
        'num_candidates': num_candidates,
        'slate_size': slate_size,
        'resample_documents': True,
        'seed': 0,
        'reward_function':clicked_quality_reward
    }

    expertEnv = environment.Environment(
        ExpertModel(slate_size),
        EEVideoSampler(1),
        num_candidates,
        slate_size,
        resample_documents=True)


    states = load('./datasets_states/states.npy', allow_pickle=True)
    policy_ = load('./datasets_states/policy.npy', allow_pickle=True)

    lts_gym_env = recsim_gym.RecSimGymEnv(expertEnv, clicked_evaluation_reward)
    recsim_gym_env = interest_evolution.create_environment(env_config)
    #agent = full_slate_q_agent.FullSlateQAgent(tf.Session(config=tf.ConfigProto(allow_soft_placement=True)),
    #                                                          recsim_gym_env.observation_space, recsim_gym_env.action_space)

    agent_irl = InverseRLAgent(recsim_gym_env, states, policy_, num_cand=num_candidates,
                               slate_size=slate_size, max_steps_per_episode=100)

    max_episode = 5
    results = []
    for i in range(max_episode):
        steps, watch, time_, q = run_one_episode(recsim_gym_env, agent_irl)
        results += ["episode "+str(i)+", total_steps: "+ str(steps) +", total_watch_time: "+
                    str(watch)+", time_episode: "+ str(time_) + ", total qual: ",str(q)]

    for i in results:
        print(i)
