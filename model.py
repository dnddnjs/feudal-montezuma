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

		self.fc_value1 = nn.Linear(288, 50)
		self.fc_value2 = nn.Linear(50, 1)

	def forward(self, inputs):
		x, (hx, cx) = inputs
		x = F.relu(self.fc(x))
		state = x
		hx, cx = self.lstm(x, (hx, cx))

		goal = hx
		value = F.relu(self.fc_value1(goal))
		value = self.fc_value2(value)

		goal_norm = torch.norm(goal, p=2, dim=1)
		goal = goal.div(goal_norm.detach())
		return goal, (hx, cx), value, state


class Worker(nn.Module):
	def __init__(self, num_outputs):
		self.num_outputs = num_outputs
		super(Worker, self).__init__()

		self.lstm = nn.LSTMCell(288, hidden_size=288)
		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)

		self.embed_size = 16
		self.fc = nn.Linear(288, self.embed_size)

		self.fc_value1 = nn.Linear(288, 50)
		self.fc_value2 = nn.Linear(50, 1)

	def forward(self, inputs):
		x, (hx, cx), goals = inputs
		hx, cx = self.lstm(x, (hx, cx))

		value = F.relu(self.fc_value1(hx))
		value = self.fc_value2(value)

		worker_embed = hx.view(hx.size(0), 
			                   self.num_outputs, 
			                   self.embed_size)

		goals = goals.sum(dim=-1)
		goal_embed = self.fc(goals)
		goal_embed = goal_embed.unsqueeze(-1)
		
		policy = torch.bmm(worker_embed, goal_embed)
		policy = policy.squeeze(-1)
		policy = F.softmax(policy)
		return policy, (hx, cx), value


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

	def forward(self, x, m_lstm, w_lstm, goals):
		percept_z = self.percept(x)
		
		m_inputs = (percept_z, m_lstm)
		goal, m_lstm, m_value, m_state = self.manager(m_inputs)
		goals = torch.cat([goal.unsqueeze(-1), goals], dim=-1)

		if goals.size(-1) > 10:
			goals = goals[:, :, -10:]

		w_inputs = (percept_z, w_lstm, goals)
		policy, w_lstm, w_value = self.worker(w_inputs)
		return policy, goal, goals, m_lstm, w_lstm, m_value, w_value, m_state