import gym
import torch
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
        state = pre_process(state)
        state = torch.Tensor(state)
        self.history = torch.stack((state, state, state, state))

    def run(self):
        super(EnvWorker, self).run()

        episode = 0
        steps = 0
        score = 0
        life = 3
        dead = False

        while True:
            if self.render:
                self.env.render()

            action = self.child_conn.recv()
            next_state, reward, done, info = self.env.step(action)
            if life > info['ale.lives']:
                dead = True
                life = info['ale.lives']

            next_state = torch.Tensor(pre_process(next_state))
            next_state = next_state.unsqueeze(0)
            self.history = torch.cat((next_state, self.history[:-1]), dim=0)

            steps += 1
            score += reward

            self.child_conn.send([self.history, reward, dead, done])

            if done and dead:
                print('{} episode | score: {:2f} | steps: {}'.format(
                    episode, score, steps
                ))
                episode += 1
                steps = 0
                score = 0
                dead = False
                life = 3
                self.init_state()

            if dead:
                dead = False
                self.init_state()

