import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'policy', 'value', 'action', 'reward', 'mask'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, state, policy, value, action, reward, mask):
        self.memory.append(Transition(state, policy, value, action, reward, mask))

    def sample(self):
        transitions = self.memory
        batch = Transition(*zip(*transitions))
        return batch


    def __len__(self):
        return len(self.memory)