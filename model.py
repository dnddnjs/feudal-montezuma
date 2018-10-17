import torch
import torch.nn as nn
import torch.nn.functional as F


class Manager(nn.Module):
	def __init__(self):
		super(Manager, self).__init__()
		self.fc = nn.Linear(288, 288)

		self.lstm = nn.LSTMCell(288, hidden_size=288)
		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)


	def forward(self, inputs):
		x, (hx, cx) = inputs
		x = F.relu(self.fc(x))
		hx, cx = self.lstm(x, (hx, cx))
		goal = hx
		goal_norm = torch.norm(goal, p=2, dim=1)
		goal /= goal_norm.detach()
		return goal, (hx, cx)


class Worker(nn.Module):
	def __init__(self, num_outputs):
		self.num_outputs = num_outputs
		super(Worker, self).__init__()

		self.lstm = nn.LSTMCell(288, hidden_size=288)
		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)

		self.embed_size = 16
		self.fc = nn.Linear(288, self.embed_size)

	def forward(self, inputs):
		x, (hx, cx), goals = inputs
		hx, cx = self.lstm(x, (hx, cx))
		worker_embed = hx.view(hx.size(0), 
			                   self.num_outputs, 
			                   self.embed_size)

		goals = goals.sum(dim=-1)
		goal_embed = self.fc(goals)
		goal_embed = goal_embed.unsqueeze(-1)
		
		policy = torch.bmm(worker_embed, goal_embed)
		policy = policy.squeeze(-1)
		policy = F.softmax(policy)
		return policy, (hx, cx)


class Percept(nn.Module):
	def __init__(self):
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
		self.fc = nn.Linear(32*9*9 ,288)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.view(x.size(0), -1)
		out = F.relu(self.fc(x))
		return out


class FuN(nn.Module):
	def __init__(self, num_outputs):
		super(FuN, self).__init__()
		self.percept = Percept()
		self.manager = Manager()
		self.worker = Worker(num_outputs)

	def forward(self, x, manager_states, worker_states, goals):
		percept_z = self.percept(x)
		
		manager_inputs = (percept_z, manager_states)
		goal, manager_states = self.manager(manager_inputs)
		goal = goal.unsqueeze(-1)
		goals = torch.cat([goal, goals], dim=-1)
		if goals.size(-1) > 10:
			goals = goals[:, :, -10:]

		worker_inputs = (percept_z, worker_states, goals)
		policy, worker_states = self.worker(worker_inputs)
		return policy, goals, manager_states, worker_states