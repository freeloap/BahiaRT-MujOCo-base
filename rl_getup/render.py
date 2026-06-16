"""可视化看 RL 起身策略：开 MuJoCo 窗口，循环"随机倒姿→起身"。

用法：rl_getup/.venv_rl/bin/python rl_getup/render.py [--model rl_getup/getup_best.zip] [--fallen]
  --fallen  只从倒地起步（默认含参考态/蹲姿起步，更易看全流程）
按 Ctrl+C 结束。
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from getup_env import GetUpEnv, CTRL_DT

from stable_baselines3 import PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "getup_best.zip"))
    ap.add_argument("--fallen", action="store_true", help="只从倒地起步")
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = GetUpEnv(seed=int(time.time()) % 100000)
    env.total = 10 ** 12          # 关闭训练助力，看真实(部署)效果
    reset_opt = {"fallen": True} if args.fallen else None

    print("打开窗口中… 循环播放 倒姿->起身。关掉窗口或 Ctrl+C 结束。")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        ep = 0
        while viewer.is_running():
            ep += 1
            obs, _ = env.reset(options=reset_opt)
            stood = False
            for _ in range(env.max_steps):
                if not viewer.is_running():
                    break
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                viewer.sync()
                time.sleep(CTRL_DT)        # 实时播放
                if info["h"] > 0.55 and info["upright"] > 0.9:
                    stood = True
            print(f"  第{ep}次: {'✅站起' if stood else '❌未站起'}  末高={info['h']:.2f}m 直立={info['upright']:+.2f}")
            time.sleep(0.6)


if __name__ == "__main__":
    main()
