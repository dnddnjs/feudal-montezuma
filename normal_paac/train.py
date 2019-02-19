import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from utils import get_entropy, get_logprob, get_grad_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gae(rewards, masks, values, last_values, args):
    returns = torch.zeros_like(rewards).to(device)
    advants = torch.zeros_like(rewards).to(device)

    running_returns = last_values.squeeze(-1)
    previous_value = last_values.squeeze(-1)
    running_advants = torch.zeros_like(running_returns)

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + args.gamma * running_returns * masks[t]
        running_tderror = rewards[t] + args.gamma * previous_value * masks[t] - \
                          values.data[t]
        running_advants = running_tderror + args.gamma * args.lamda * \
                          running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def surrogate_loss(net, advant, history, old_logit, action):
    new_logit, new_value = net(history)
    
    new_policy = F.log_softmax(new_logit, dim=-1)
    action = action.unsqueeze(-1)

    new_policy = new_policy.gather(-1, action).squeeze(-1)
    old_policy = F.log_softmax(old_logit, dim=-1)
    old_policy = old_policy.gather(-1, action).squeeze(-1)

    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advant
    return surrogate, ratio, new_logit, new_value.squeeze(-1)


def train_model(net, optimizer, transition, last_values, args):
    histories = torch.stack(transition.history)
    old_logits = torch.stack(transition.policy)
    old_values = torch.stack(transition.value).squeeze(-1)
    actions = torch.Tensor(transition.action).long().to(device)
    rewards = torch.Tensor(transition.reward).to(device)
    masks = torch.Tensor(transition.mask).to(device)
    
    returns, advants = get_gae(rewards, masks, old_values, last_values, args)

    histories = histories.view(-1, histories.size(2), histories.size(3), histories.size(4))
    old_logits = old_logits.view(-1, old_logits.size(2))
    returns = returns.view(-1)
    advants = advants.view(-1)
    actions = actions.view(-1)
    old_values = old_values.view(-1)
    rewards = rewards.view(-1)
    masks = masks.view(-1)
    
    n = histories.size(0)
    arr = np.arange(n)
    criterion = nn.MSELoss()

    entropy = 0
    grad_norm = 0
    for epoch in range(3):
        # print('epoch is ' + str(epoch))
        np.random.shuffle(arr)

        for i in range(n // args.batch_size):
            batch_index = arr[args.batch_size * i: args.batch_size * (i + 1)]
            batch_index = torch.Tensor(batch_index).long().to(device)
            inputs = histories[batch_index]
            returns_samples = returns[batch_index]
            advants_samples = advants[batch_index]
            actions_samples = actions[batch_index]
            old_logits_samples = old_logits[batch_index].detach()
            old_value_samples = old_values[batch_index].detach()

            loss, ratio, logits, values = surrogate_loss(net, advants_samples, inputs,
                                                         old_logits_samples, actions_samples)
            
            clipped_values = old_value_samples + \
                             torch.clamp(values - old_value_samples,
                                         -args.clip_param,
                                         args.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples.detach())
            critic_loss2 = criterion(values, returns_samples.detach())
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()
            
            policies = F.softmax(logits, dim=-1)
            log_policies = F.log_softmax(logits, dim=-1)
            
            entropy = - (policies * log_policies).mean()
            loss = actor_loss + critic_loss - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grad_norm = get_grad_norm(net)
            optimizer.step()

    return entropy, grad_norm