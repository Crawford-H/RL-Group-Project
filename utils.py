import gymnasium as gym
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
)


def make_env(domain_randomize=False, render_mode="rgb_array"):
    env = gym.make(
        "CarRacing-v3", domain_randomize=domain_randomize, render_mode=render_mode
    )
    env = gym.wrappers.ResizeObservation(env, (64, 64))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    return env


def make_vec_env(num_envs=1):
    vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    return vec_env


def make_eval_env():
    env = DummyVecEnv([make_env])
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)
    return env
