import os
import sys
import gym
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import pre_process, get_action
from model import FuN
from memory import Memory
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="MontezumaRevenge-v4", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--gamma', default=0.99, help='')
parser.add_argument('--batch_size', default=32, help='')
parser.add_argument('--initial_exploration', default=1000, help='')
parser.add_argument('--update_target', default=10000, help='')
parser.add_argument('--log_interval', default=1, help='')
parser.add_argument('--goal_score', default=300, help='')
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(net, target_net, optimizer, batch):
    history = torch.stack(batch.history).to(device)
    next_history = torch.stack(batch.next_history).to(device)
    actions = torch.Tensor(batch.action).long().to(device)
    rewards = torch.Tensor(batch.reward).to(device)
    masks = torch.Tensor(batch.mask).to(device)

    pred = net(history).squeeze(1)
    next_pred = target_net(next_history).squeeze(1)
    one_hot_action = torch.zeros(args.batch_size, pred.size(-1))
    one_hot_action = one_hot_action.to(device)
    one_hot_action.scatter_(1, actions.unsqueeze(1), 1)
    pred = torch.sum(pred.mul(one_hot_action), dim=1)
    target = rewards + args.gamma * next_pred.max(1)[0] * masks
    
    loss = F.smooth_l1_loss(pred, target.detach(), size_average=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.cpu().data


def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    img_shape = env.observation_space.shape
    num_actions = env.action_space.n
    print('image size:', img_shape)
    print('action size:', num_actions)

    net = FuN(num_actions)

    optimizer = optim.RMSprop(net.parameters(), lr=0.00025, eps=0.01)
    writer = SummaryWriter('logs')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    net.to(device)
    net.train()

    memory = Memory(100000)
    epsilon = 1.0
    steps = 0
    
    for e in range(10000):
        done = False
        dead = False

        score = 0
        avg_loss = []
        start_life = 5
        state = env.reset()

        state = pre_process(state)
        state = torch.Tensor(state).to(device)
        state = state.permute(2, 0, 1)

        manager_hx = torch.zeros(1, 288).to(device)
        manager_cx = torch.zeros(1, 288).to(device)
        manager_states = (manager_hx, manager_cx)

        worker_hx = torch.zeros(1, 288).to(device)
        worker_cx = torch.zeros(1, 288).to(device)
        worker_states = (worker_hx, worker_cx)

        goals = torch.zeros(1, 288, 1).to(device)

        while not done:
            if args.render:
                env.render()

            steps += 1
            net_output = net(state.unsqueeze(0), manager_states, worker_states, goals)
            policy, goal, manager_states, worker_states = net_output
            action = get_action(policy, num_actions)
            next_state, reward, done, info = env.step(action)
            
            next_state = pre_process(next_state)
            next_state = torch.Tensor(next_state).to(device)
            next_state = next_state.permute(2, 0, 1)

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']
            
            score += reward
            reward = np.clip(reward, -1, 1)

            mask = 0 if dead else 1
            
            if dead:
                dead = False
                manager_hx = torch.zeros(1, 288).to(device)
                manager_cx = torch.zeros(1, 288).to(device)
                manager_states = (manager_hx, manager_cx)

                worker_hx = torch.zeros(1, 288).to(device)
                worker_cx = torch.zeros(1, 288).to(device)
                worker_states = (worker_hx, worker_cx)

                goals = torch.zeros(1, 288, 1).to(device)
                
            state = next_state


        if e % args.log_interval == 0:
            print('{} episode | score: {:.2f} | steps: {} | loss: {:.4f}'.format(
                e, score, steps, np.mean(avg_loss)))
            writer.add_scalar('log/score', float(score), steps)
            writer.add_scalar('log/score', np.mean(avg_loss), steps)

        if score > args.goal_score:
            ckpt_path = args.save_path + 'model.pth'
            torch.save(net.state_dict(), ckpt_path)
            print('running score exceeds 400 so end')
            break

if __name__=="__main__":
    main()