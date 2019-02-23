import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Change code to deal with multiple environments input

class Manager(nn.Module):
    def __init__(self, dilation, num_actions, args, device):
        super(Manager, self).__init__()

        hidden_size = 128

        self.hx_memory = [torch.zeros(args.num_envs, num_actions * 16).to(device) for _ in range(dilation)]

        self.cx_memory = [torch.zeros(args.num_envs, num_actions * 16).to(device) for _ in range(dilation)]
        self.hidden_size = hidden_size
        self.horizon = dilation
        self.index = 0

        self.fc = nn.Linear(num_actions * 16, num_actions * 16)
        # todo: change lstm to dilated lstm
        self.lstm = nn.LSTMCell(num_actions * 16, hidden_size=num_actions * 16)
        # todo: add lstm initialization
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.fc_critic1 = nn.Linear(num_actions * 16, 50)
        self.fc_critic2 = nn.Linear(50, 1)

        self.fc_actor = nn.Linear(50, num_actions)

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = F.relu(self.fc(x))
        state = x

        hx_t_1 = self.hx_memory[self.index]
        cx_t_1 = self.cx_memory[self.index]
        self.hx_memory[self.index] = hx
        self.cx_memory[self.index] = cx

        hx, cx = self.lstm(x, (hx_t_1, cx_t_1))
        self.index += 1
        if self.index >= self.horizon:
            self.index %= self.horizon

        goal = cx
        value = F.relu(self.fc_critic1(goal))
        value = self.fc_critic2(value)
        
        goal_norm = torch.norm(goal, p=2, dim=1).unsqueeze(1)
        goal = goal / goal_norm.detach()
        return goal, (hx, cx), value, state

class Worker(nn.Module):
    def __init__(self, num_actions):
        self.num_actions = num_actions
        super(Worker, self).__init__()

        self.lstm = nn.LSTMCell(num_actions * 16, hidden_size=num_actions * 16)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # Linear projection of goal has no bias
        self.fc = nn.Linear(num_actions * 16, 16, bias=False)

        self.fc_critic1 = nn.Linear(num_actions * 16, 50)
        self.fc_critic1_out = nn.Linear(50, 1)
        
        self.fc_critic2 = nn.Linear(num_actions * 16, 50)
        self.fc_critic2_out = nn.Linear(50, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs):
        x, (hx, cx), goals = inputs
        hx, cx = self.lstm(x, (hx, cx))

        value_ext = F.relu(self.fc_critic1(hx))
        value_ext = self.fc_critic1_out(value_ext)
        
        value_int = F.relu(self.fc_critic2(hx))
        value_int = self.fc_critic2_out(value_int)

        worker_embed = hx.view(hx.size(0),
                               self.num_actions,
                               16)

        goals = goals.sum(dim=1)
        # goals should be disconnected from Manager.
        goal_embed = self.fc(goals.detach())
        goal_embed = goal_embed.unsqueeze(-1)

        policy = torch.bmm(worker_embed, goal_embed)
        policy = policy.squeeze(-1)
        policy = F.softmax(policy, dim=-1)
        return policy, (hx, cx), value_ext, value_int

class Percept(nn.Module):
    def __init__(self, num_actions):
        super(Percept, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=8,
            stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2)
        self.fc = nn.Linear(32 * 9 * 9, num_actions * 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        out = F.relu(self.fc(x))
        return out


class FuN(nn.Module):
    def __init__(self, num_actions, args, device):
        super(FuN, self).__init__()
        self.horizon = args.horizon
        self.num_envs = args.num_envs
        self.device = device

        self.percept = Percept(num_actions)
        self.manager = Manager(self.horizon, num_actions, args, device)
        self.worker = Worker(num_actions)


    def forward(self, x, m_lstm, w_lstm, goals_horizon):
        percept_z = self.percept(x)

        m_inputs = (percept_z, m_lstm)
        goal, m_lstm, m_value, m_state = self.manager(m_inputs)
        
        # todo: at the start, there is no previous goals. Need to be checked
        goals_horizon = torch.cat([goals_horizon[:, 1:], goal.unsqueeze(1)], dim=1)
        
        w_inputs = (percept_z, w_lstm, goals_horizon)
        policy, w_lstm, w_value_ext, w_value_int = self.worker(w_inputs)
        return policy, goal, goals_horizon, m_lstm, w_lstm, m_value, w_value_ext, w_value_int, m_state
