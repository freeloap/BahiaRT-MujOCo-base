"""把训练好的 PPO 策略导出为 onnx，供比赛 agent 用 onnxruntime 加载。

用法：rl_getup/.venv_rl/bin/python rl_getup/export_onnx.py
输入：rl_getup/getup_ppo.zip
输出：mujococodebase/skills/rl/get_up.onnx  （输入 (1,75) float32 -> 输出 (1,23) 动作）
部署侧（GetUpRL 技能）按相同的观测构造与动作映射使用它。
"""
import os
import sys

import torch
from stable_baselines3 import PPO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "getup_ppo.zip")
DST = os.path.join(ROOT, "mujococodebase", "skills", "rl", "get_up.onnx")


class OnnxablePolicy(torch.nn.Module):
    """只保留「观测 -> 确定性动作」的前向，便于 onnxruntime 推理。"""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        return self.policy._predict(observation, deterministic=True)


def main():
    if not os.path.exists(SRC):
        sys.exit(f"找不到 {SRC}，请先训练（train.py）。")
    model = PPO.load(SRC, device="cpu")
    wrapper = OnnxablePolicy(model.policy).eval()

    obs_dim = model.observation_space.shape[0]
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, DST,
        input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=13,
    )
    print(f"已导出 onnx: {DST}  (obs_dim={obs_dim})")


if __name__ == "__main__":
    main()
