import os
import sys
import gym
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import FuN
from memory import Memory
from train import train_model
from utils import pre_process, get_action

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Enduro-v4", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--m_gamma', default=0.99, help='')
parser.add_argument('--w_gamma', default=0.95, help='')
parser.add_argument('--log_interval', default=1, help='')
parser.add_argument('--goal_score', default=300, help='')
parser.add_argument('--horizon', default=1, help='')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    img_shape = env.observation_space.shape
    num_actions = env.action_space.n
    print('image size:', img_shape)
    print('action size:', num_actions)

    net = FuN(num_actions, args.horizon)

    optimizer = optim.RMSprop(net.parameters(), lr=0.00025, eps=0.01)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    net.to(device)
    net.train()
    
    total_steps = 0
    memory = Memory(capacity=400)

    for e in range(10000):    
        done = False
        steps = 0
        score = 0
        
        avg_loss = []
        state = env.reset()

        state = pre_process(state)
        state = torch.Tensor(state).to(device)
        state = state.permute(2, 0, 1)

        m_hx = torch.zeros(1, num_actions*16).to(device)
        m_cx = torch.zeros(1, num_actions*16).to(device)
        m_lstm = (m_hx, m_cx)

        w_hx = torch.zeros(1, num_actions*16).to(device)
        w_cx = torch.zeros(1, num_actions*16).to(device)
        w_lstm = (w_hx, w_cx)

        goals = torch.zeros(1, num_actions*16, 1).to(device)

        while True:
            if args.render:
                env.render()
            steps += 1
            total_steps += 1
            
            net_output = net(state.unsqueeze(0), m_lstm, w_lstm, goals)
            policy, goal, goals, m_lstm, w_lstm, m_value, w_value, m_state = net_output

            action = get_action(policy, num_actions)
            next_state, reward, done, _ = env.step(action)

            next_state = pre_process(next_state)
            next_state = torch.Tensor(next_state).to(device)
            next_state = next_state.permute(2, 0, 1)
            
            score += reward
            reward = np.clip(reward, -1, 1)

            mask = 0 if done else 1

            memory.push(action, reward, mask, goal, policy,
                        m_lstm, w_lstm, m_value, w_value, m_state)
            state = next_state
            
            if done or steps == 400:
                batch = memory.sample()
                loss = train_model(net, optimizer, batch, args.w_gamma, args.m_gamma, args.horizon)
                avg_loss.append(loss.cpu().data)
                break

        if e % args.log_interval == 0:
            print('{} episode | score: {:.2f} | steps: {} | loss: {:.4f}'.format(
                e, score, total_steps, np.mean(avg_loss)))

        if score > args.goal_score:
            ckpt_path = args.save_path + 'model.pth'
            torch.save(net.state_dict(), ckpt_path)
            print('running score exceeds 400 so end')
            break

if __name__=="__main__":
    main()