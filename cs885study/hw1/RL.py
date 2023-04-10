import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''
        #the solution is not perfect yet. (23/04/09)
        import random
        Q = initialQ
        policy = np.zeros(self.mdp.nStates,int)

        for i in range(nEpisodes):
            visited = np.zeros([mdp.nActions,mdp.nStates])
            state = s0
            for step in range(nSteps):
                if random.uniform(0, 1) < epsilon:
                    action =  np.random.randint(0, mdp.nActions)
                else:

                    policy = np.argmax(Q,0)
                    weight = np.exp(Q[:,state] / temperature)
                    weight /= np.sum(weight)
                    #print("weight: ", weight, "Q", Q[:,state])
                    action_space = list(range(mdp.nActions))
                    action = np.random.choice(action_space, mdp.nActions, p=weight)
                    
                visited[action, state] +=1

                alpha = 1 / visited[action, state]
                reward, state_next = self.sampleRewardAndNextState(state,action)

                td_target = reward + mdp.discount * np.max(Q[:,state_next])
                Q_delta = alpha * (td_target - Q[action,state])
                Q[action,state] += Q_delta
                Q[action,state_next] += Q_delta
                state = state_next
                
                #print(policy)

        return [Q,policy]  