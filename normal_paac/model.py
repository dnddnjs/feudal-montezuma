import torch.nn.functional as F
import torch.nn as nn
import torch


class ActorCritic(nn.Module):
    def __init__(self, num_outputs):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32,
                               kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=2)
        self.fc = nn.Linear(4 * 4 * 64, 512)
        self.fc_actor = nn.Linear(512, num_outputs)
        self.fc_critic = nn.Linear(512, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.relu(x)
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value