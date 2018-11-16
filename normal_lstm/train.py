import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_returns(rewards, masks, gamma):
    returns = torch.zeros_like(rewards)
    running_returns = 0
    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        returns[t] = running_returns

    if returns.std() != 0:
        returns = (returns - returns.mean()) / returns.std()
    return returns


def train_model(net, optimizer, batch, m_gamma, w_gamma, horizon):
    actions = torch.Tensor(batch.action).long().to(device)
    rewards = torch.Tensor(batch.reward).to(device)
    masks = torch.Tensor(batch.mask).to(device)
    goals = torch.stack(batch.goal).to(device)
    policies = torch.stack(batch.policy).to(device)
    m_states = torch.stack(batch.m_state).to(device)
    m_values = torch.stack(batch.m_value).to(device)
    w_values = torch.stack(batch.w_value).to(device)

    batch_size = len(actions)
    
    m_returns = get_returns(rewards, masks, m_gamma)
    w_returns = get_returns(rewards, masks, w_gamma)
    
    rewards_int = torch.zeros_like(rewards).to(device)
    # todo: how to get intrinsic reward before 10 steps

    for i in range(horizon, len(rewards)):
        cos_sum = 0
        for j in range(1, horizon+1):
            alpha = m_states[i] - m_states[i-j]
            beta = goals[i-j]
            cosine_sim = F.cosine_similarity(alpha, beta)
            cos_sum = cos_sum + cosine_sim
        reward_int = cos_sum / horizon
        rewards_int[i] = reward_int.detach()
    returns_int = get_returns(rewards_int, masks, w_gamma)

    m_loss = torch.zeros_like(w_returns).to(device)
    w_loss = torch.zeros_like(m_returns).to(device)
    # todo: how to update manager near end state
    for i in range(horizon, len(rewards)-horizon):
        m_advantage = m_returns[i] - m_values[i]
        alpha = m_states[i + horizon] - m_states[i]
        beta = goals[i]
        cosine_sim = F.cosine_similarity(alpha.detach(), beta)
        m_loss[i] = - m_advantage * cosine_sim

        log_policy = torch.log(policies[i] + 1e-5)
        w_advantage = w_returns[i] + returns_int[i] - w_values[i]
        w_loss[i] = - w_advantage * log_policy[:, actions[i]]

    m_loss = m_loss.sum()
    w_loss = w_loss.sum()
    m_loss_value = F.mse_loss(m_values.squeeze(), m_returns)
    w_loss_value = F.mse_loss(w_values.squeeze(), w_returns + returns_int)
    loss = w_loss + w_loss_value + m_loss + m_loss_value
    loss = loss / batch_size
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss