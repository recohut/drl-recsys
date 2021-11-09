from recsim.agents import full_slate_q_agent, random_agent
from rl import *
from utils import *
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

    expertEnv = environment.Environment(
        ExpertModel(slate_size),
        EEVideoSampler(0),
        num_candidates,
        slate_size,
        resample_documents=True)

    lts_gym_env = recsim_gym.RecSimGymEnv(expertEnv, clicked_evaluation_reward)
    recsim_gym_env = interest_evolution.create_environment(env_config)
    observation_0 = recsim_gym_env.reset()

    for i in range(3):
        recommendation_slate_0 = [0,1]
        print(recommendation_slate_0)
        observation_1, reward, done, scores, _ = recsim_gym_env.step(recommendation_slate_0)
        print('Observation ' + str(i))
        print('Available documents')
        doc_strings = ['doc_id ' + key + str(value) for key, value
                   in observation_1['doc'].items()]
        print('\n'.join(doc_strings))
        rsp_strings = [str(response) for response in observation_1['response']]
        print('User responses to documents in the slate')
        print('\n'.join(rsp_strings))
        print('Reward: ', reward)
        print("User observation noise:", observation_1['user'][0], " interests features: ", observation_1['user'][1:])
        print("*******************************************")
