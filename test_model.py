import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy


def make_env():
    env = gym.make("CarRacing-v3", render_mode="human")
    env = gym.wrappers.ResizeObservation(env, (64, 64))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    return env


if __name__ == "__main__":
    model_path = "sac_carracing_v3.zip"

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)

    model = SAC.load(model_path, env=env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    exit()
