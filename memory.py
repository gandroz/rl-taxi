import numpy as np
from collections import deque

class Experience():
    def __init__(self, dim):
        self.states = np.zeros(dim, dtype=np.uint8)
        self.actions = np.zeros(dim, dtype=np.uint8)
        self.rewards = np.zeros(dim, dtype=np.int8)
        self.new_states = np.zeros(dim, dtype=np.uint8)
        self.done = np.zeros(dim, dtype=np.uint8)
        self._idx = 0
    
    def append(self, state, action, reward, new_state, done):
        self.states[self._idx] = state
        self.actions[self._idx] = action
        self.rewards[self._idx] = reward
        self.new_states[self._idx] = new_state
        self.done[self._idx] = 1 if done else 0
        self._idx += 1
    
    
class Memory():
    def __init__(self, max_len:int=0, rng=None, seed:int=None):
        assert max_len is not None and max_len > 0, "max_len must be given to constructor"
        self.states = deque(maxlen=max_len)
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.new_states = deque(maxlen=max_len)
        self.done = deque(maxlen=max_len)
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng
        
    def sample(self, batch_size):
        batch_idxs = self.rng.choice(np.arange(self.length), batch_size, replace=False)
        experience = Experience(batch_size)
        S = list(self.states)
        A = list(self.actions)
        R = list(self.rewards)
        NS = list(self.new_states)
        D = list(self.done)
        for idx in batch_idxs:
            experience.append(state=S[idx], action=A[idx], reward=R[idx],
                              new_state=NS[idx], done=D[idx])
        return experience
    
    def append(self, state, action, reward, new_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.done.append(done)
        
    @property
    def length(self):
        return len(self.states)