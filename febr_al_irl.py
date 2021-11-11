from maxEnt_irl import *
from utils import *
import sys
import random
from numpy import save
import calendar;
import time;

if __name__ == '__main__':
    

    # simulation parameters
    
    # number of categories of video topics
    NUM_FEATURES = int(sys.argv[1])
    slate_size = int(sys.argv[2])
    num_candidates = int(sys.argv[3])
    steps = int(sys.argv[4])
    num_trajectories = int(sys.argv[5])
    num_expert = int(sys.argv[6])
    # timestamp for files naming
    ts = calendar.timegm(time.gmtime())

    env_config = {
        'num_candidates': num_candidates,
        'slate_size': slate_size,
        'resample_documents': True,
        'seed': 0,
    }

    f_handle = open('./datasets_states/states'+str(ts)+'.npy', 'wb')
    traj = []
    states = []
    for expert in range(num_expert):
        topic_id_vector = random.randint(0, NUM_FEATURES-1)
        print("expert ",expert,"topic: ", topic_id_vector)
        expertEnv = environment.Environment(
            ExpertModel(slate_size),
            EEVideoSampler(topic_id_vector),
            num_candidates,
            slate_size,
            resample_documents=True)
        r = generate_trajectories(expertEnv, num_trajectories, steps, num_candidates, slate_size)

        traj = traj + r[0]
        states = states + r[1]
    save(f_handle, states)

    f_handle.close()
    print("finishing generating trajectories")
    print("num_trajs= ",len(traj)," len states ", len(states))
    rewards, policy = maxEnt_irl(states, 0.9, traj, rl_algo="value_iter", candidate=num_candidates, slate=slate_size, n_steps=steps)
    
    # save to npy files
    save('./datasets_states/rewards'+str(ts)+'.npy', rewards)
    print("rewards optimisation finished")
    save('./datasets_states/policy'+str(ts)+'.npy', policy)
    print("policy optimisation finished")