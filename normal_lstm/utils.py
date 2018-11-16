import torch
import random
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


def pre_process(image):
    image = np.array(image)
    image = resize(image, (84, 84, 3))
    # image = rgb2gray(image)
    return image


def get_action(policy, num_actions):
    policy = policy.data.cpu().numpy()[0]
    action = np.random.choice(num_actions, 1, p=policy)[0]
    return action