import gym
import torch
import numpy as np
from copy import deepcopy
from utils import pre_process
from torch.multiprocessing import Process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnvWorker(Process):
    def __init__(self, env_name, render, child_conn):
        super(EnvWorker, self).__init__()
        self.env = gym.make(env_name)
        self.render = render
        self.child_conn = child_conn
        self.init_state()

    def init_state(self):
        state = self.env.reset()
        
        state, _, _, _ = self.env.step(1)
        state = pre_process(state)
        self.history = np.moveaxis(state, -1, 0)

    def run(self):
        super(EnvWorker, self).run()

        episode = 0
        steps = 0
        score = 0
        life = 5
        dead = False

        while True:
            if self.render:
                self.env.render()

            action = self.child_conn.recv()
            next_state, reward, done, info = self.env.step(action + 1)
            
            if life > info['ale.lives']:
                dead = True
                life = info['ale.lives']

            next_state = pre_process(next_state)
            self.history = np.moveaxis(next_state, -1, 0)

            steps += 1
            score += reward

            self.child_conn.send([deepcopy(self.history), reward, dead, done])

            if done and dead:
                # print('{} episode | score: {:2f} | steps: {}'.format(
                #     episode, score, steps
                # ))
                episode += 1
                steps = 0
                score = 0
                dead = False
                life = 5
                self.init_state()

            if dead:
                dead = False
                self.init_state()

