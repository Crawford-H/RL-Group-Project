import sys

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from utils import make_eval_env, make_vec_env

N_ENVS = 20
OUT_DIR = "./logs/sac"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_envs = int(sys.argv[1])
    else:
        num_envs = N_ENVS

    train_env = make_vec_env(num_envs=num_envs)
    eval_env = make_vec_env(num_envs=2)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=OUT_DIR,
        log_path=OUT_DIR,
        eval_freq=25_000 // num_envs,
        render=False,
        deterministic=True,
        n_eval_episodes=10,
    )

    model = SAC(
        "CnnPolicy",
        train_env,
        verbose=0,
        buffer_size=100_000,
        learning_starts=10_000,
        tensorboard_log=OUT_DIR,
    )

    model.learn(
        total_timesteps=4_000_000,
        log_interval=4,
        callback=eval_callback,
        progress_bar=True,
    )
    model.save("sac_carracing_v3")
