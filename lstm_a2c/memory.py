import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition',
                        ('history', 'next_history', 'action', 'reward',
                         'mask', 'goal', 'policy', 'm_lstm', 'w_lstm',
                         'm_value', 'w_value_ext', 'w_value_int', 'm_state'))


class Memory(object):
    def __init__(self):
        self.memory = []
        self.position = 0

    def push(self, history, next_history, action, reward,
             mask, goal, policy, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state):
        """Saves a transition."""
        self.memory.append(Transition(history, next_history, action, reward, mask,
                           goal, policy, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state))

    def sample(self):
        transitions = Transition(*zip(*self.memory))
        return transitions

    def __len__(self):
        return len(self.memory)