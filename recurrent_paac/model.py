import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CLSTM(nn.Module):
    def __init__(self, num_actions, lstm_size):
        super(A2CLSTM, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        self.fc = nn.Linear(32 * 6 * 6, lstm_size)
        self.lstm  = nn.LSTMCell(lstm_size, lstm_size)
        
        self.fc_actor = nn.Linear(lstm_size, num_actions)
        self.fc_critic = nn.Linear(lstm_size, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()
                
            if isinstance(p, nn.LSTMCell):
                p.bias_ih.data.zero_()
                p.bias_hh.data.zero_()

    def forward(self, inputs):
        x, (hx, cx) = inputs
        
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value, (hx, cx)