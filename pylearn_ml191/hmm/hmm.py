import numpy as np

class HMM(object):
    """ Multinomial Hidden Markov Model """
    def __init__(self, A, B, pi):
        self.A = A 
        self.B = B
        self.pi = pi

    def forward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)
        #
        F = np.zeros((N,T))
        F[:, 0] = self.pi * self.B[:, obs_seq[0]]
        #
        for t in range(1, T):
            F[:, t] = (F[:, t-1] @ self.A) * self.B[:, obs_seq[t]]
            
        return F

    def backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)
        #
        X = np.zeros((N, T))
        X[:, -1:] = 1
        #
        for t in reversed(range(T-1)):
            X[:, t] = np.sum(X[:, t+1] * self.A * self.B[:, obs_seq[t+1]], axis=1)
        return X

    def score(self, obs_seq):
        """ Sol. for problem 1: Calculate probability for any observation sequence with specify model"""
        return np.sum(self.forward(obs_seq)[:, -1])
    
    def decode(self, obs_seq):
        """ Sol. for problem 2: Viterbi algo """
        N = self.A.shape[0]
        T = len(obs_seq)
        #
        V = np.zeros((N, T))
        B_ptr = np.zeros((N, T))
        #
        V[:, 0] = self.pi * self.B[:, obs_seq[0]]
        B_ptr[:, 0] = 0
        #
        for t in range(1, T):
            for n in range(N):
                seq_probs = V[:, t-1] * self.A[:, n] * self.B[n, obs_seq[t]]
                B_ptr[n, t] = np.argmax(V[:, t-1] * self.A[:, n])
                V[n, t] = np.max(seq_probs)
        
        # Get state seq
        states_idx = np.argmax(V, axis=0)
        prob = np.max(V, axis=0)[-1]
        
        for i in reversed(range(1, T)):
            states_idx[i - 1] = B_ptr[states_idx[i], i]
            
        return states_idx, prob
    
    def _sample_from(self, probs):
        return np.where(np.random.multinomial(1, probs) == 1)[0][0]
    
    def sample(self, T, random_state=21):
        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        #
        states[0] = self._sample_from(self.pi)
        observations[0] = self._sample_from(self.B[states[0], :])
        for t in range(1, T):
            states[t] = self._sample_from(self.A[states[t - 1], :])
            observations[t] = self._sample_from(self.B[states[t], :])
        return observations, states

    def fit(self, observations, max_step=100, stop_criterion=0.01, verbose=1):
        """ Sol. for problem 3: Learning model """
        # N states
        N = self.A.shape[0] 
        # T observations
        T = len(observations)
        
        step = 0
        while True:
            step += 1 
            #
            alpha = self.forward(observations) 
            beta = self.backward(observations)
            #
            xi = np.zeros((N, N, T - 1))
            for t in range(T - 1):
                denom = ((alpha[:, t] @ self.A) * self.B[:, observations[t + 1]]) @ beta[:, t + 1]
                for i in range(N):
                    numer = alpha[i, t] * self.A[i, :] * self.B[:, observations[t + 1]] * beta[:, t + 1]
                    xi[i, : , t] = numer / denom
            #
            gamma = np.squeeze(np.sum(xi, axis=1))
            prod =  (alpha[:, T - 1] * beta[:, T - 1]).reshape((-1, 1))
            gamma = np.hstack((gamma,  prod / np.sum(prod))) 
            #
            pi = gamma[:, 0]
            A = np.sum(xi, 2) / np.sum(gamma[:, :-1], axis=1).reshape((-1, 1)) 
            B = np.copy(self.B)
            #
            num_obs_vals = self.B.shape[1]
            sumgamma = np.sum(gamma, axis=1)
            for v in range(num_obs_vals):
                mask = observations == v
                B[:, v] = np.sum(gamma[:, mask], axis=1) / sumgamma
            
            # delta: sum of difference of each entries between two iteration ==> check stop
            delta = np.sum(np.fabs(np.array((
                *(A-self.A).flatten(),
                *(B-self.B).flatten(),
                *(pi-self.pi)))))

            if step % verbose == 0:
                print("Iter: {}, delta: {:0.5f} ".format(step, delta))
            
            self.A = A
            self.B = B
            self.pi = pi
            
            if delta < stop_criterion or step == max_step:
                break
                
    def show_model(self):
        np.set_printoptions(precision=4, suppress=True)
        print('A: Transition probability matrix')
        print(self.A)
        print('------------------------------')
        print('B: Emission probability matrix')
        print(self.B)
        print('-------------------------------')
        print('pi: Initital state distribution')
        print(self.pi)