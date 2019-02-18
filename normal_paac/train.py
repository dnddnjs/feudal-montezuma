import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import get_entropy, get_logprob, get_grad_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_return(rewards, masks, values, last_values, args):
    returns = torch.zeros_like(rewards).to(device)
    running_returns = last_values.squeeze(-1)

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + args.gamma * running_returns * masks[t]
        returns[t] = running_returns

    return returns


def train_model(net, optimizer, transition, last_values, args):
    logits = torch.stack(transition.policy)
    values = torch.stack(transition.value).squeeze(-1)
    actions = torch.Tensor(transition.action).long().to(device)
    rewards = torch.Tensor(transition.reward).to(device)
    masks = torch.Tensor(transition.mask).to(device)

    returns = get_return(rewards, masks, values, last_values, args)

    # get policy gradient
    advantage = (returns - values).detach()

    policies = F.softmax(logits, dim=-1)
    log_policies = F.log_softmax(logits, dim=-1)
    action_log_policies = log_policies.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    
    mseloss = nn.MSELoss()
    loss_v = mseloss(values, returns.detach())
    loss_p = - (advantage * action_log_policies).mean()
    entropy = - (policies * log_policies).mean()
    loss = loss_p + args.value_coef * loss_v - args.entropy_coef * entropy
    
    optimizer.zero_grad()
    loss.backward()
    grad_norm = get_grad_norm(net)
    # torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)
    optimizer.step()
    # print(loss_v.item(), loss_p.item(), entropy.item())
    return entropy, grad_norm