import gym
import torch
import random
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
        # state, _, _, _ = self.env.step(1)    
        state = pre_process(state)
        self.state = np.moveaxis(state, -1, 0)

    def run(self):
        super(EnvWorker, self).run()

        episode = 0
        steps = 0
        score = 0

        while True:
            if self.render:
                self.env.render()

            action = self.child_conn.recv()
            next_state, reward, done, _ = self.env.step(action)
            next_state = pre_process(next_state)
   
            steps += 1
            score += reward
            
            self.child_conn.send([deepcopy(self.state), reward, done])
            self.state = np.moveaxis(next_state, -1, 0)   

            if done:
                episode += 1
                steps = 0
                score = 0
                self.init_state()

                


            