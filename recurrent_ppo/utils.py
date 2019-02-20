import torch
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from torch.distributions import Categorical


def get_action(logits, num_actions):
    m = Categorical(logits=logits)
    actions = m.sample()
    actions = actions.data.cpu().numpy()
    return actions


def get_entropy(logits):
    m = Categorical(logits=logits)
    entropy = m.entropy()
    return entropy


def get_logprob(logits, actions):
    m = Categorical(logits=logits)
    log_probs = m.log_prob(actions)
    return log_probs


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    grad_norm = total_norm ** (1. / 2)
    return grad_norm
