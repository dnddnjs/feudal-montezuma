import os
import gym
import time
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from env import make_env, RenderSubprocVecEnv

from model import ActorCritic
from utils import get_action, pre_process
from train import train_model
from env import EnvWorker
from memory import Memory

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="PongNoFrameskip-v0", help='')
parser.add_argument('--seed', default=7000, help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--gamma', default=0.99, help='')
parser.add_argument('--lamda', default=0.95, help='')
parser.add_argument('--log_interval', default=10, help='')
parser.add_argument('--save_interval', default=1000, help='')
parser.add_argument('--num_envs', default=12, help='')
parser.add_argument('--num_step', default=20, help='')
parser.add_argument('--value_coef', default=0.5, help='')
parser.add_argument('--entropy_coef', default=0.01, help='')
parser.add_argument('--lr', default=7e-4, help='')
parser.add_argument('--eps', default=1e-5, help='')
parser.add_argument('--clip_grad_norm', default=3, help='')
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print('==> make {} environment'.format(args.num_envs))
    # reference: https://github.com/lnpalmer/A2C/blob/master/train.py
    env_fns = []
    for rank in range(args.num_envs):
        env_fns.append(lambda: make_env(args.env_name, 
                                        rank, args.seed + rank))
    if args.render:
        venv = RenderSubprocVecEnv(env_fns, args.render_interval)
    else:
        venv = SubprocVecEnv(env_fns)
    venv = VecFrameStack(venv, 4)
    
    num_inputs = venv.observation_space.shape
    num_actions = venv.action_space.n - 1
    print('state size:', num_inputs)
    print('action size:', num_actions)
    
    print('==> make actor critic network')
    net = ActorCritic(num_actions)
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, eps=args.eps)
    writer = SummaryWriter('logs')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    net.to(device)
    net.train()
    
    score = np.zeros(args.num_envs)
    global_steps, count, batch_time = 0, 0, 0
    
    histories = venv.reset()
    histories = histories.astype(float) / 236.
    histories = np.transpose(histories, [0, 3, 1, 2])
    
    while True:
        count += 1
        memory = Memory()
        global_steps += (args.num_envs * args.num_step)
        
        start = time.time()
        # gather samples from environment
        for _ in range(args.num_step):
            policies, values = net(torch.Tensor(histories).to(device))
            actions = get_action(policies, num_actions)
            
            next_histories, rewards, dones, _ = venv.step(actions)
            masks = 1 - dones
            next_histories = next_histories.astype(float) / 236.
            next_histories = np.transpose(next_histories, [0, 3, 1, 2])
            score += np.array(rewards)
            
            rewards = np.hstack(rewards)
            masks = np.hstack(masks)
            memory.push(torch.Tensor(next_histories).to(device), 
                        policies, values, actions, rewards, masks)
            histories = next_histories
            
            
            for i in range(args.num_envs): 
                if dones[i]:
                    entropy = - policies * torch.log(policies + 1e-5)
                    entropy = entropy.mean().data.cpu()
                    print('global steps {} | score: {} | entropy: {:.4f} | batch time: {:.3f}'.format(
                        global_steps, score[i], entropy, batch_time))

                    writer.add_scalar('log/score', score[i], global_steps)
            score *= masks
            
        # train network with accumulated samples
        transitions = memory.sample()
        train_model(net, optimizer, transitions, args)
        
        end = time.time()
        batch_time = end - start
        
        if count % args.save_interval == 0:
            ckpt_path = args.save_path + 'model.pt'
            torch.save(net.state_dict(), ckpt_path)


if __name__=="__main__":
    main()
    