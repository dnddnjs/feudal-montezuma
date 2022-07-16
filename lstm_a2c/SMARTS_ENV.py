import numpy as np
from pathlib import Path
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.custom_observations import *
from smarts.zoo.agent_spec import AgentSpec
from FUNRL.lstm_a2c.utils import pre_process
from smarts.env.hiway_env import HiWayEnv

import gym

"Define observation/action space range of values"

ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
)

OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)

def observation_adapter(env_observation):


    return lane_ttc_observation_adapter.transform(env_observation)


def reward_adapter(env_obs, env_reward):
    return env_reward


def action_adapter(model_action):
    throttle, brake, steering = model_action
    return np.array([throttle, brake, steering * np.pi * 0.25])

"Define FuN agent"
class FuNAgent(Agent):
    def __init__(self, path_to_model, observation_space):
        path_to_model = str(path_to_model)

    def act(self, obs: Observation, **configs):
        desired_speed = 10
        lane_idx = 1

        trajectory_points = min(10, len(obs.waypoint_paths[lane_idx]))

        trajectory = [
            [
                obs.waypoint_paths[lane_idx][i].pos[0]
                for i in range(trajectory_points)
            ],
            [
                obs.waypoint_paths[lane_idx][i].pos[1]
                for i in range(trajectory_points)
            ],
            [
                obs.waypoint_paths[lane_idx][i].heading
                for i in range(trajectory_points)
            ],
            [desired_speed for i in range(trajectory_points)],
        ]
        return trajectory






#Make environment
scenario_path ='scenarios/intersections/4lane'

Fun_agent = {
    "agent_spec": AgentSpec(
        interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=500),
        agent_params={
            "path_to_model": Path(__file__).resolve().parent / "model",
            "observation_space": OBSERVATION_SPACE,
        },
        agent_builder=FuNAgent,
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
        action_adapter=action_adapter,
    ),
    "observation_space": OBSERVATION_SPACE,
    "action_space": ACTION_SPACE,
}






#Reset env and build agent


#Step Env



#close env






