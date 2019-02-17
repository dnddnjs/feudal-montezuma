import torch
import torch.nn.functional as F

from utils import get_entropy, get_logprob, get_grad_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_return(rewards, masks, values, last_values, args):
    returns = torch.zeros_like(rewards).to(device)
    running_returns = last_values.squeeze(-1)

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + args.gamma * running_returns * masks[t]
        returns[t] = running_returns

    return returns


def train_model(net, optimizer, transition, args):
    next_histories = torch.stack(transition.next_history)
    logits = torch.stack(transition.policy)
    values = torch.stack(transition.value).squeeze(-1)
    actions = torch.Tensor(transition.action).long().to(device)
    rewards = torch.Tensor(transition.reward).to(device)
    masks = torch.Tensor(transition.mask).to(device)

    _, last_values = net(next_histories[-1])
    returns = get_return(rewards, masks, values, last_values, args)

    # get policy gradient
    advantage = returns - values

    # log_policies = torch.log(policies.gather(-1, actions.unsqueeze(-1)) + 1e-5)
    # log_policies = log_policies.squeeze(-1)
    
    log_policies = get_logprob(logits, actions)
    loss_p = - (advantage.detach() * log_policies).mean()
    loss_v = advantage.pow(2).mean()
    entropy = get_entropy(logits).mean()
    loss = loss_p + args.value_coef * loss_v - args.entropy_coef * entropy
    
    optimizer.zero_grad()
    loss.backward()
    print(get_grad_norm(net))
    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)
    optimizer.step()
    # print(loss_v.item(), loss_p.item(), entropy.item())
    return entropy