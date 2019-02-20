import os
import gym
import time
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from env import make_env, RenderSubprocVecEnv

from model import ActorCritic
from utils import get_action, get_entropy
from train import train_model
from memory import Memory

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="PongNoFrameskip-v0", help='')
parser.add_argument('--seed', default=7000, help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--gamma', default=0.99, help='')
parser.add_argument('--lamda', default=0.95, help='')
parser.add_argument('--clip_param', default=0.1, help='')
parser.add_argument('--batch_size', default=32*8, help='')
parser.add_argument('--hidden_size', default=512, help='')
parser.add_argument('--log_interval', default=5, help='')
parser.add_argument('--render_interval', default=4, help='')
parser.add_argument('--save_interval', default=1000, help='')
parser.add_argument('--num_envs', type=int, default=8, help='')
parser.add_argument('--num_step', type=int, default=128, help='')
parser.add_argument('--value_coef', default=0.5, help='')
parser.add_argument('--entropy_coef', default=0.01, help='')
parser.add_argument('--lr', default=2.5e-4, help='')
parser.add_argument('--eps', default=1e-5, help='')
parser.add_argument('--alpha', type=float, default=0.99, help='')
parser.add_argument('--clip_grad_norm', default=0.5, help='')
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
        venv = RenderSubprocVecEnv(env_fns)
    else:
        venv = SubprocVecEnv(env_fns)
    # venv = VecFrameStack(venv, 4)
    
    num_inputs = venv.observation_space.shape
    num_actions = venv.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)
    
    print('==> make actor critic network')
    net = ActorCritic(num_actions)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    writer = SummaryWriter('logs')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    net.to(device)
    net.train()
    
    score = np.zeros(args.num_envs)
    episode_scores = deque(maxlen=10)
    
    global_steps, count, batch_time, entropy = 0, 0, 0, 0
    
    histories = venv.reset()
    histories = np.transpose(histories, [0, 3, 1, 2])
    hx = torch.zeros(args.num_envs, args.hidden_size).to(device)
    cx = torch.zeros(args.num_envs, args.hidden_size).to(device)
    
    start = time.time()
    while True:
        count += 1
        memory = Memory()
        global_steps += (args.num_envs * args.num_step)

        # gather samples from environment
        hx = hx.detach()
        cx = cx.detach()
        for _ in range(args.num_step):
            logits, values, (hx, cx) = net(torch.Tensor(histories).to(device), hx, cx)
            actions = get_action(logits, num_actions)
            
            next_histories, rewards, dones, _ = venv.step(actions)
            masks = 1 - dones

            next_histories = np.transpose(next_histories, [0, 3, 1, 2])
            score += np.array(rewards)
            
            rewards = np.hstack(rewards)
            masks = np.hstack(masks)
            masks_t = torch.Tensor(masks).to(device)
            masks_t = masks_t.unsqueeze(-1)
            masks_t = masks_t.expand(args.num_envs, args.hidden_size)
            hx = hx * masks_t
            cx = cx * masks_t
            
            memory.push(torch.Tensor(histories).to(device), 
                        logits, values, actions, rewards, masks, hx, cx)
            histories = next_histories
            
            for i in range(args.num_envs): 
                if dones[i]:
                    episode_scores.append(score[i])
            score *= masks
            
        # train network with accumulated samples
        transitions = memory.sample()
        _, last_values, _ = net(torch.Tensor(next_histories).to(device), hx, cx)
        entropy, grad_norm = train_model(net, optimizer, transitions, last_values, args)

        if count % args.log_interval == 0:
            end = time.time()
            mean_score = np.mean(episode_scores)
            print('steps {} | mean score: {} | entropy: {:.2f} | '
                  'grad norm : {:.3f} | frame per sec: {:.0f}'.format(
                   global_steps, mean_score, entropy.item(), grad_norm, 
                   global_steps/(end-start)))
            writer.add_scalar('log/score', mean_score, global_steps)
            
        if count % args.save_interval == 0:
            ckpt_path = args.save_path + 'model.pt'
            torch.save(net.state_dict(), ckpt_path)


if __name__=="__main__":
    main()
    