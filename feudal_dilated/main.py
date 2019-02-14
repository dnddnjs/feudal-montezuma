import os
import gym
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe

from model import FuN
from utils import *
from train import train_model
from env import EnvWorker
from memory import Memory

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="BreakoutDeterministic-v4", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--m_gamma', default=0.999, help='')
parser.add_argument('--w_gamma', default=0.99, help='')
parser.add_argument('--goal_score', default=400, help='')
parser.add_argument('--log_interval', default=10, help='')
parser.add_argument('--save_interval', default=1000, help='')
parser.add_argument('--num_envs', default=16, help='')
parser.add_argument('--num_step', default=400, help='')
parser.add_argument('--value_coef', default=0.5, help='')
parser.add_argument('--entropy_coef', default=0.01, help='')
parser.add_argument('--lr', default=7e-4, help='')
parser.add_argument('--eps', default=1e-5, help='')
parser.add_argument('--horizon', default=10, help='')
parser.add_argument('--clip_grad_norm', default=5, help='')
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    img_shape = env.observation_space.shape
    num_actions = env.action_space.n - 1
    print('image size:', img_shape)
    print('action size:', num_actions)

    net = FuN(num_actions, args, device)
    optimizer = optim.RMSprop(net.parameters(), lr=0.00025, eps=0.01)
    writer = SummaryWriter('logs')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    workers = []
    parent_conns = []
    child_conns = []

    for i in range(args.num_envs):
        parent_conn, child_conn = Pipe()
        worker = EnvWorker(args.env_name, args.render, child_conn)
        worker.start()
        workers.append(worker)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    net.to(device)
    net.train()

    global_steps = 0
    score = np.zeros(args.num_envs)
    count = 0
    grad_norm = 0

    histories = torch.zeros([args.num_envs, 3, 84, 84]).to(device)
    
    m_hx = torch.zeros(args.num_envs, num_actions * 16).to(device)
    m_cx = torch.zeros(args.num_envs, num_actions * 16).to(device)
    m_lstm = (m_hx, m_cx)

    w_hx = torch.zeros(args.num_envs, num_actions * 16).to(device)
    w_cx = torch.zeros(args.num_envs, num_actions * 16).to(device)
    w_lstm = (w_hx, w_cx)

    goals_horizon = torch.zeros(args.num_envs, args.horizon + 1, num_actions * 16).to(device)

    while True:
        count += 1
        memory = Memory()
        global_steps += (args.num_envs * args.num_step)

        # gather samples from the environment
        for i in range(args.num_step):
            # TODO: think about net output
            net_output = net(histories.to(device), m_lstm, w_lstm, goals_horizon)
            policies, goal, goals_horizon, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state = net_output

            actions = get_action(policies, num_actions)

            # send action to each worker environment and get state information
            next_histories, rewards, masks, dones = [], [], [], []

            for i, (parent_conn, action) in enumerate(zip(parent_conns, actions)):
                parent_conn.send(action)
                next_history, reward, dead, done = parent_conn.recv()
                next_histories.append(next_history)
                rewards.append(reward)
                masks.append(1 - dead)
                dones.append(done)
                
                if dead:
                    m_hx_mask = torch.ones(args.num_envs, num_actions * 16).to(device)
                    m_hx_mask[i, :] = m_hx_mask[i, :]*0
                    m_cx_mask = torch.ones(args.num_envs, num_actions * 16).to(device)
                    m_cx_mask[i, :] = m_cx_mask[i, :]*0
                    m_hx, m_cx = m_lstm
                    m_hx = m_hx * m_hx_mask
                    m_cx = m_cx * m_cx_mask
                    m_lstm = (m_hx, m_cx)
                    
                    w_hx_mask = torch.ones(args.num_envs, num_actions * 16).to(device)
                    w_hx_mask[i, :] = w_hx_mask[i, :]*0
                    w_cx_mask = torch.ones(args.num_envs, num_actions * 16).to(device)
                    w_cx_mask[i, :] = w_cx_mask[i, :]*0                    
                    w_hx, w_cx = w_lstm
                    w_hx = w_hx * w_hx_mask
                    w_cx = w_cx * w_cx_mask
                    w_lstm = (w_hx, w_cx)
                    
                    goal_init = torch.zeros(args.horizon + 1, num_actions * 16).to(device)
                    goals_horizon[i] = goal_init

                    
            score += rewards[0]

            # if agent in first environment dies, print and log score
            for i in range(args.num_envs):
                if dones[i]: 
                    entropy = - policies * torch.log(policies + 1e-5)
                    entropy = entropy.mean().data.cpu()
                    print('global steps {} | score: {} | entropy: {:.4f} | grad norm: {:.3f} '.format(global_steps, score[i], entropy, grad_norm))
                    if i == 0:
                        writer.add_scalar('log/score', score[i], global_steps)
                    score[i] = 0

            next_histories = torch.Tensor(next_histories).to(device)
            rewards = np.hstack(rewards)
            masks = np.hstack(masks)
            memory.push(histories, next_histories,
                        actions, rewards, masks, goal,
                        policies, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state)
            histories = next_histories

        # Train every args.num_step
        if (global_steps % args.num_step) == 0:  # Need to fix logic
            transitions = memory.sample()
            loss, grad_norm = train_model(net, optimizer, transitions, args)
            m_hx, m_cx = m_lstm
            m_lstm = (m_hx.detach(), m_cx.detach())
            w_hx, w_cx = w_lstm
            w_lstm = (w_hx.detach(), w_cx.detach())
            goals_horizon = goals_horizon.detach()
            # avg_loss.append(loss.cpu().data)

        if count % args.save_interval == 0:
            ckpt_path = args.save_path + 'model.pt'
            torch.save(net.state_dict(), ckpt_path)


if __name__=="__main__":
    main()














