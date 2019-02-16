import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(net, optimizer, transition, args):
    next_histories = torch.stack(transition.next_history)
    policies = torch.stack(transition.policy)
    values = torch.stack(transition.value)
    actions = torch.Tensor(transition.action).long().to(device)
    rewards = torch.Tensor(transition.reward).to(device)
    masks = torch.Tensor(transition.mask).to(device)

    entropy = - policies * torch.log(policies + 1e-5)
    policies = policies.gather(-1, actions.unsqueeze(-1))
    log_policies = torch.log(policies.squeeze() + 1e-5)
    last_policies, last_values = net(next_histories[-1])        

    # get multi-step td-error
    returns = torch.zeros_like(rewards).to(device)
    running_returns = last_values.squeeze(-1)
    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + args.gamma * running_returns * masks[t]
        returns[t] = running_returns
        
    td_errors = (returns - values.squeeze(-1)).detach()

    # get policy gradient
    loss_p = - log_policies * td_errors
    loss_v= F.mse_loss(values.squeeze(-1), returns.detach())
    loss = loss_p.mean() + args.value_coef * loss_v.mean() - args.entropy_coef * entropy.detach().mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)
    optimizer.step()
    return entropy