import gym
import torch
import random
import numpy as np
from copy import deepcopy
from utils import pre_process
from torch.multiprocessing import Process

import numpy as np
import gym
from gym import spaces
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnv
from multiprocessing import Process, Pipe


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
        self.history = np.stack((state, state, state, state), axis=0)

    def run(self):
        super(EnvWorker, self).run()

        episode = 0
        steps = 0
        score = 0

        while True:
            if self.render:
                self.env.render()

            action = self.child_conn.recv()
            next_state, reward, done, info = self.env.step(action+1)
            
            next_state = pre_process(next_state)
            next_state = np.reshape([next_state], (1, 84, 84))
            self.history = np.append(self.history[1:, :, :], next_state, axis=0)          

            steps += 1
            score += reward
            
            self.child_conn.send([deepcopy(self.history), reward, done])

            if done:
                episode += 1
                steps = 0
                score = 0
                self.init_state()
                
            # if dead:
            #     dead = False
            #     self.init_state()


def make_env(env_name, rank, seed):
    env = make_atari(env_name)
    env.seed(seed + rank)
    env = wrap_deepmind(env)
    return env


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        elif cmd == 'render':
            env.render()
        else:
            raise NotImplementedError


class RenderSubprocVecEnv(VecEnv):
    def __init__(self, env_fns, render_interval):
        """ Minor addition to SubprocVecEnv, automatically renders environments
        envs: list of gym environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

        self.render_interval = render_interval
        self.render_timer = 0

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)

        self.render_timer += 1
        if self.render_timer == self.render_interval:
            for remote in self.remotes:
                remote.send(('render', None))
            self.render_timer = 0

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)



            