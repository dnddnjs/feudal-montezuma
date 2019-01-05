import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_returns(rewards, masks, gamma, values):
    returns = torch.zeros_like(rewards)
    running_returns = values[-1].squeeze()

    for t in reversed(range(0, len(rewards)-1)):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        returns[t] = running_returns

    if returns.std() != 0:
        returns = (returns - returns.mean()) / returns.std()

    return returns


def train_model(net, optimizer, transition, args):

    # policy, m_lstm, w_lstm, m_value, w_value, m_state
    # histories = torch.stack(transition.history).to(device)
    # next_histories = torch.stack(transition.next_history).to(device)
    actions = torch.Tensor(transition.action).long().to(device)
    rewards = torch.Tensor(transition.reward).to(device)
    masks = torch.Tensor(transition.mask).to(device)
    goals = torch.stack(transition.goal).to(device)
    policies = torch.stack(transition.policy).to(device)
    m_states = torch.stack(transition.m_state).to(device)
    m_values = torch.stack(transition.m_value).to(device)
    w_values_ext = torch.stack(transition.w_value_ext).to(device)
    w_values_int = torch.stack(transition.w_value_int).to(device)
    # m_lstms = torch.stack(transition.m_lstm).to(device)
    # w_lstms = torch.stack(transition.w_lstm).to(device)
    
    transition_size = len(actions)

    # policies, values = policies[0], w_values[0]  # net(histories[0])
    # last_values = w_values[-1]
    # _, last_values = net(next_histories[-1])
    
    m_returns = get_returns(rewards, masks, args.m_gamma, m_values)
    w_returns = get_returns(rewards, masks, args.w_gamma, w_values_ext)

    intrinsic_rewards = torch.zeros_like(rewards).to(device)
    # todo: how to get intrinsic reward before 10 steps

    for i in range(args.horizon, len(rewards)):
        cos_sum = 0
        for j in range(1, args.horizon + 1):
            alpha = m_states[i] - m_states[i - j]
            beta = goals[i - j]
            cosine_sim = F.cosine_similarity(alpha, beta)
            cos_sum = cos_sum + cosine_sim
        intrinsic_reward = cos_sum / args.horizon
        intrinsic_rewards[i] = intrinsic_reward.detach()
    returns_int = get_returns(intrinsic_rewards, masks, args.w_gamma, w_values_int)

    m_loss = torch.zeros_like(w_returns).to(device)
    w_loss = torch.zeros_like(m_returns).to(device)

    # todo: how to update manager near end state
    for i in range(args.horizon, len(rewards)-args.horizon):
        m_advantage = m_returns[i] - m_values[i].squeeze(-1)
        alpha = m_states[i + args.horizon] - m_states[i]
        beta = goals[i]
        cosine_sim = F.cosine_similarity(alpha.detach(), beta)
        m_loss[i] = - m_advantage * cosine_sim

        log_policy = torch.log(policies[i] + 1e-5)
        w_advantage = w_returns[i] + returns_int[i] - w_values_ext[i].squeeze(-1) - w_values_int[i].squeeze(-1)
        log_policy = log_policy.gather(-1, actions[i].unsqueeze(-1))
        w_loss[i] = - w_advantage * log_policy.squeeze(-1)

    m_loss = m_loss.sum()
    w_loss = w_loss.sum()
    m_loss_value = F.mse_loss(m_values.squeeze(), m_returns)
    w_loss_value_ext = F.mse_loss(w_values_ext.squeeze(), w_returns)
    w_loss_value_int = F.mse_loss(w_values_int.squeeze(), returns_int)
    loss = w_loss_value_int
    # loss = w_loss + w_loss_value_ext + w_loss_value_int + m_loss + m_loss_value
    loss = loss / transition_size
    # TODO: Add entropy to loss for exploration

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm(net.parameters(), args.clip_grad_norm)
    optimizer.step()
    # return loss