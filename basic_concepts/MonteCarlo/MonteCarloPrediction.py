import gymnasium as gym
import math
import numpy as np
from collections import defaultdict
#reference code for episode generator: https://marcinbogdanski.github.io/rl-sketchpad/RL_An_Introduction_2018/0501_First_Visit_MC_Prediction.html


INF = math.inf
num_bins = [16, 16, 16, 16]

# Define the range of values for each observation variable
#position, velocity, angle, ang_velo
obs_range = [(-4.8, 4.8), (-INF, INF), (-.418, .418), (-INF, INF)]

def generate_episode(env, policy):
    """Generete one complete episode.
    
    Returns:
        trajectory: list of tuples [(st, rew, done, act), (...), (...)],
                    where St can be e.g tuple of ints or anything really
        T: index of terminal state, NOT length of trajectory
    """
    trajectory = []
    done = True
    obs, _ = env.reset()
    while True:
        # === time step starts here ===
        At = policy(obs)
        obs, Rt, terminated, truncated, info = env.step(At)
        trajectory.append((obs, Rt, terminated, At))
        if terminated:  
            break
        # === time step ends here ===
    return trajectory, len(trajectory)-1


# Define a function to discretize the observation space
def discretize(obs):
    # Convert each observation variable to a discrete value
    obs_discrete = []
    for i in range(len(obs)):
        obs_i = obs[i]
        obs_i_min, obs_i_max = obs_range[i]
        obs_i_bins = np.linspace(obs_i_min, obs_i_max, num_bins[i] - 1)
        obs_discrete.append(np.digitize(obs_i, obs_i_bins))
    return obs_discrete
        
        
def first_visit_MC_prediction(env, policy, ep, gamma):
    """First Visit MC Prediction
    Params:
        env - environment
        policy - function in a form: policy(state)->action
        ep - number of episodes to run
        gamma - discount factor
    """
    V = dict()
    Returns = defaultdict(list)    # dict of lists
        
    for _ in range(ep):
        traj, T = generate_episode(env, policy)
        G = 0
        for t in range(T-1,-1,-1):
            St, _, _, _ = traj[t]      # (st, rew, done, act)
            _, Rt_1, _, _ = traj[t+1]
            
            G = gamma * G + Rt_1
            #not sure this discretizing idea is correct. (23/03/11)
            # To-Do: need some optimization?
            past_St_discrete = [discretize(traj[i][0].tolist()) for i in range(0, t)]
            St_discrete = discretize(St)
            str_St = str(St_discrete) #make hashable
            
            if not St_discrete in past_St_discrete:
                Returns[str_St].append(G)
                V[str_St] = np.average(Returns[str_St])
    
    return V

def policy_rd(St):
    # Some Non-sense random policy
    position, velocity, angle, ang_velo = St
    if -2<= position <=2 and -0.1<=angle<=angle:
        return 0  
    else:
        return 1  
    
def get_policy(type_policy = 'rd1'):
    if type_policy == 'rd1':
        return policy_rd
    


if __name__ == '__main__':
    GAMMA = 0.7
    NUM_EPISODE = 1000
    policy = get_policy()
    env = gym.make('CartPole-v1')
    V = first_visit_MC_prediction(env, policy, ep=NUM_EPISODE, gamma=GAMMA)
    print("Estimated Valuefunction is equal to: \n" ,  V)
    

