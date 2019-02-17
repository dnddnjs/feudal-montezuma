import torch.nn.functional as F
import torch.nn as nn
import torch


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
    
class ActorCritic(nn.Module):
    def __init__(self, num_outputs):
        super(ActorCritic, self).__init__()
        hidden_size = 512
        
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)
        
        self.actor_linear = init_(nn.Linear(hidden_size, num_outputs))
        
        '''
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32,
                               kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=2)
        self.fc = nn.Linear(4 * 4 * 64, 512)
        self.fc_actor = nn.Linear(512, num_outputs)
        self.fc_critic = nn.Linear(512, 1)
        '''

    def forward(self, x):
        x = self.main(x / 255.)
        logit = self.actor_linear(x)
        value = self.critic_linear(x)
        return logit, value