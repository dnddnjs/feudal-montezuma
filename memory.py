import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('action', 'reward', 'mask', 'goal', 'policy', 'm_lstm', 'w_lstm', 'm_value', 'w_value', 'm_state'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, action, reward, mask, goal, policy, m_lstm, w_lstm, m_value, w_value, m_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(action, reward, mask, goal, policy, m_lstm, w_lstm, m_value, w_value, m_state))
        self.memory[self.position] = Transition(action, reward, mask, goal, policy, m_lstm, w_lstm, m_value, w_value, m_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)