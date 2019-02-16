import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gae(rewards, masks, values, last_values, args):
    returns = torch.zeros_like(rewards).to(device)
    advants = torch.zeros_like(rewards).to(device)

    running_returns = last_values.squeeze(-1)
    previous_value = torch.zeros(args.num_envs).to(device)
    running_advants = torch.zeros(args.num_envs).to(device)

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + args.gamma * running_returns * masks[t]
        running_tderror = rewards[t] + args.gamma * previous_value * masks[t] - values.data[t]
        running_advants = running_tderror + args.gamma * args.lamda * running_advants * masks[t]
        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def train_model(net, optimizer, transition, args):
    next_histories = torch.stack(transition.next_history)
    policies = torch.stack(transition.policy)
    values = torch.stack(transition.value).squeeze(-1)
    actions = torch.Tensor(transition.action).long().to(device)
    rewards = torch.Tensor(transition.reward).to(device)
    masks = torch.Tensor(transition.mask).to(device)

    entropy = - policies * torch.log(policies + 1e-5)
    policies = policies.gather(-1, actions.unsqueeze(-1))
    log_policies = torch.log(policies.squeeze() + 1e-5)
    _, last_values = net(next_histories[-1])

    returns, advants = get_gae(rewards, masks, values, last_values, args)

    # get policy gradient
    loss_p = - log_policies * advants
    loss_v= F.mse_loss(values, returns)
    loss = loss_p.mean() + args.value_coef * loss_v.mean() - args.entropy_coef * entropy.mean()
    
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)
    optimizer.step()
    return entropy