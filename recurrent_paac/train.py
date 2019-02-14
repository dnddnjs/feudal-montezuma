import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(net, optimizer, transition, args):
    states = torch.stack(transition.state)
    policies = torch.stack(transition.policy)
    values = torch.stack(transition.value).squeeze(-1)
    actions = torch.Tensor(transition.action).long().to(device)
    rewards = torch.Tensor(transition.reward).to(device)
    masks = torch.Tensor(transition.mask).to(device)

    entropy = - policies * torch.log(policies + 1e-5)
    policies = policies.gather(-1, actions.unsqueeze(-1))
    log_policies = torch.log(policies.squeeze(-1) + 1e-5)       

    # get multi-step td-error
    returns = torch.zeros_like(rewards).to(device)
    returns[-1] = values[-1].squeeze(-1)
    running_returns = values[-1].squeeze(-1)

    for t in reversed(range(0, len(rewards)-1)):
        running_returns = rewards[t] + args.gamma * running_returns * masks[t]
        returns[t] = running_returns
        
    td_errors = (returns - values).detach()

    # get policy gradient
    loss_p = - log_policies * td_errors
    loss_v= F.mse_loss(values, returns.detach())
    loss = loss_p.mean() + args.value_coef * loss_v - args.entropy_coef * entropy.detach().mean()
    # print(loss_p.mean().item(), loss_v.item(), entropy.mean().item())
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)
    optimizer.step()
    return entropy