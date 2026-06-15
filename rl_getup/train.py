"""用 PPO 训练 T1 起身策略（stable-baselines3）。

无 GPU 时用 CPU 多环境并行训练，较慢但可行。先小步数验证奖励上升，再加大步数。
用法：
    rl_getup/.venv_rl/bin/python rl_getup/train.py --steps 5000000 --n-envs 8
产物：rl_getup/getup_ppo.zip（策略），随后用 export_onnx.py 导出。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from getup_env import GetUpEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env(rank):
    def _f():
        return GetUpEnv(seed=rank)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5_000_000, help="总训练步数")
    ap.add_argument("--n-envs", type=int, default=8, help="并行环境数")
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "getup_ppo"))
    args = ap.parse_args()

    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    env = VecMonitor(env)  # 记录每回合回报，使日志出现 rollout/ep_rew_mean
    model = PPO(
        "MlpPolicy", env,
        n_steps=2048, batch_size=2048, gae_lambda=0.95, gamma=0.99,
        learning_rate=3e-4, ent_coef=0.01, n_epochs=5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
    )
    ckpt = CheckpointCallback(save_freq=max(1, 200_000 // args.n_envs),
                              save_path=os.path.dirname(args.out), name_prefix="getup_ckpt")
    model.learn(total_timesteps=args.steps, callback=ckpt)
    model.save(args.out)
    print(f"已保存策略: {args.out}.zip")
    print("下一步：导出 onnx -> rl_getup/export_onnx.py")


if __name__ == "__main__":
    main()
