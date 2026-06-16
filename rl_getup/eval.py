"""评估训练好的起身策略：多回合确定性回放，统计最终躯干高度/直立度/成功率。

用法：rl_getup/.venv_rl/bin/python rl_getup/eval.py [--model rl_getup/getup_ppo.zip] [--episodes 20]
成功判定：回合末躯干高度 > 0.5m 且直立度 > 0.9。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from getup_env import GetUpEnv, STAND_HEIGHT

from stable_baselines3 import PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "getup_ppo.zip"))
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = GetUpEnv(seed=12345)
    env.total = 10 ** 12   # 关闭训练用的垂直助力，测真实(部署=无助力)起身能力

    ok = 0
    hs, ups = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(options={"fallen": True})  # 只从倒地起步，测真起身
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
        h, up = info["h"], info["upright"]
        hs.append(h); ups.append(up)
        success = h > 0.5 and up > 0.9
        ok += int(success)
        print(f"  回合{ep:2d}: 末高={h:.3f}m 直立={up:+.2f} {'✅站起' if success else '❌'}")

    print(f"\n成功率 {ok}/{args.episodes} = {100*ok/args.episodes:.0f}%  "
          f"平均末高={np.mean(hs):.3f}m 平均直立={np.mean(ups):+.2f}  (站立目标高 {STAND_HEIGHT})")


if __name__ == "__main__":
    main()
