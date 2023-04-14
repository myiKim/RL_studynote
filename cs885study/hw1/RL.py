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
        Q = initialQ
        policy = np.zeros(self.mdp.nStates,int)
        #visited = np.zeros([mdp.nActions,mdp.nStates])
        for i in range(nEpisodes):
            visited = np.zeros([mdp.nActions,mdp.nStates]) #here or there?
            state = s0
            for step in range(nSteps):
                if temperature==0:
                    #action =  np.random.randint(0, mdp.nActions)
                    action = np.argmax(Q, axis=0)[state]
                else:
                    if np.random.uniform(0, 1) < epsilon:
                        action =  np.random.randint(0, mdp.nActions)
                    else:

                        policy = np.argmax(Q,0)
                        weight = np.exp(Q[:,state] / temperature)
                        weight /= np.sum(weight)
                        #print("weight: ", weight, "Q", Q[:,state])
                        action_space = list(range(mdp.nActions))
                        action = np.random.choice(action_space, mdp.nActions, p=weight)                    

                #observe s' and r 
                reward, state_next = self.sampleRewardAndNextState(state,action) 
                #update counts n(s,a) <- n(s,a) + 1
                visited[action, state] +=1
                # defining learning rate alpha <- a/n(s,a)
                alpha = 1.0 / visited[action, state]
                
                #update Q-value
                td_target = reward + mdp.discount * np.max(Q[:,state_next])
                Q_delta = alpha * (td_target - Q[action,state])
                Q[action,state] += Q_delta
                Q[action][state] = np.round(Q[action][state], 2)
                # s <- s'
                state = state_next
                
        return [Q,policy]  