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


def train_model(net, optimizer, batch, gamma):
    actions = torch.Tensor(batch.action).long().to(device)
    rewards = torch.Tensor(batch.reward).to(device)
    masks = torch.Tensor(batch.mask).to(device)
    goals = torch.stack(batch.goal).to(device)
    policies = torch.stack(batch.policy).to(device)
    m_states = torch.stack(batch.m_state).to(device)
    m_values = torch.stack(batch.m_value).to(device)
    w_values = torch.stack(batch.w_value).to(device)

    returns = get_returns(rewards, masks, gamma)
    
    
    rewards_int = torch.zeros_like(rewards).to(device)
    for i in range(len(rewards)):
        if i > 10:
            cos_sum = 0
            for j in range(10):
                alpha = m_states[i] - m_states[i - j]
                beta = goals[j]
                cosine_sim = F.cosine_similarity(alpha.detach(), beta)
                cos_sum = cos_sum + cosine_sim
            reward_int = cos_sum / 10
        else:
            reward_int = 0
            # cos_sum = 0
            # for j in range(i):
            #     alpha = m_states[i] - m_states[i-j]
            #     beta = goals[j]
            #     cosine_sim = F.cosine_similarity(alpha.detach(), beta)
            #     cos_sum = cos_sum + cosine_sim
            # reward_int = cos_sum / i
        rewards_int[i] = reward_int
    returns_int = get_returns(rewards_int, masks, gamma)

    loss1 = torch.zeros_like(returns).to(device)
    loss3 = torch.zeros_like(returns).to(device)
    for i in range(len(rewards)-10):
        m_advantage = returns[i] - m_values[i]
        alpha = m_states[i + 10] - m_states[i]
        beta = goals[i]
        cosine_sim = F.cosine_similarity(alpha.detach(), beta)
        loss1[i] = - m_advantage.detach() * cosine_sim

        log_policy = torch.log(policies[i] + 1e-5)
        w_advantage = returns[i] + returns_int[i] - w_values[i]
        loss3[i] = -w_advantage.detach() * log_policy[:, actions[i]]


    loss1 = loss1.sum()
    loss3 = loss3.sum()
    loss2 = F.mse_loss(m_values.squeeze(), returns)
    loss4 = F.mse_loss(w_values.squeeze(), returns + returns_int)
    loss = loss1 + loss2 + loss3 + loss4
    loss = loss / len(rewards)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    

    return loss