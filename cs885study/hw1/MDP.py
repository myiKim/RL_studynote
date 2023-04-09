import numpy as np
import numpy.linalg as LA
class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        epsilon = None       
        V = np.max(self.R, axis=0)
        iterId = 0
        while True:
            iterId +=1
            Valist = []
            for i in range(self.nActions):
                Va= self.R[i]+ self.discount * np.dot(self.T[i] ,V)
                Valist.append(Va)
            V, pastV = np.max(Valist,0), V
            epsilon =  LA.norm(V-pastV, ord= np.inf) 
            #Stopping condition.
            if epsilon < tolerance or iterId > nIterations:
                break            

        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        Valist = []
        for i in range(self.nActions):
            Va= self.R[i]+ self.discount * np.dot(self.T[i] ,V)
            Valist.append(Va)
        policy = np.argmax(Valist,0)

        return policy

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        statespace = list(range(self.nStates))
        
        Ta = self.T[policy,statespace]
        Ra = self.R[policy,statespace]
        A = np.identity(len(Ta)) - (self.discount * Ta)
        # V = LA.solve(A, Ra)
        V = np.matmul(np.linalg.inv(A), Ra)
        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        policy = initialPolicy.astype(int)
        V = np.zeros(self.nStates)
        iterId = 0
        while True:
            #evaluate policy
            V = self.evaluatePolicy(policy)
            #improve policy
            policy, past_policy = self.extractPolicy(V), policy
            if all([p==pp for p, pp in zip(policy, past_policy)]):                
                break
            iterId+=1

        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        V = initialV
        iterId = 0
        statespace = list(range(self.nStates))
        Tpi = T[policy,statespace]
        Rpi = R[policy,statespace]       

        while True:
            V, pastV =  Rpi + (self.discount * np.matmul(Tpi, V)), V            
            epsilon =  LA.norm(V-pastV, ord= np.inf)                
            if epsilon < tolerance or iterId>nIterations:
                break
            iterId +=1          

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        policy = initialPolicy
        V = initialV
        iterId = 0
        
        while True:            
            #Partially Evaluate the given policy
            Vnew, _, _  = self.evaluatePolicyPartially1(policy, V, nIterations=nEvalIterations, tolerance=0.01)

            #induce the policy from the estimated V (record the policy)
            policy = self.extractPolicy1(Vnew)
            epsilon = max(abs(Vnew - V))
            V = Vnew

            if epsilon < tolerance or iterId>nIterations:
                break      
            iterId += 1

        return [policy,V,iterId,epsilon]
        