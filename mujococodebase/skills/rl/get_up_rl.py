"""基于强化学习策略的起身技能（GetUpRL）。

加载 rl_getup 训练并导出的 get_up.onnx，按与训练环境 **完全一致** 的观测构造和
动作映射运行，使倒地的机器人靠学习到的反馈策略站起来。

观测(75) = 关节角归一化(23) + 关节速*0.1(23) + 躯干重力投影(3) + 躯干角速度*0.25(3) + 上一步动作(23)
动作(23) = 每关节 [-1,1] 线性映射到该关节限位（度），再按 KP/KD 下发（须与训练环境一致）。

⚠️ 仅当 get_up.onnx 存在时才可用（训练→导出后）。未训练时不要注册到 SkillsManager。
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

from mujococodebase.skills.skill import Skill
from mujococodebase.utils.neural_network import load_network, run_network

# 必须与 rl_getup/getup_env.py 中的常量一致
KP, KD = 200.0, 5.0
MOTORS = ["he1","he2","lae1","lae2","lae3","lae4","rae1","rae2","rae3","rae4","te1",
          "lle1","lle2","lle3","lle4","lle5","lle6","rle1","rle2","rle3","rle4","rle5","rle6"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "get_up.onnx")


class GetUpRL(Skill):
    UPRIGHT_DONE = 0.92     # 重力投影 -z 分量超过此值视为基本直立
    HEIGHT_DONE = 0.5       # 躯干高度阈值（m）
    STABLE_CYCLES = 10      # 连续满足判定的周期数

    def __init__(self, agent):
        super().__init__(agent)
        self.model = load_network(model_path=MODEL_PATH)
        # 关节限位（弧度），顺序同 MOTORS；来自 T1.JOINT_LIMITS（度）
        lim = self.agent.robot.JOINT_LIMITS
        self.jlo = np.radians([lim[m][0] for m in MOTORS])
        self.jhi = np.radians([lim[m][1] for m in MOTORS])
        # 动作幅度，须与 getup_env 一致：action=0->0弧度(站立)、±1触及限位
        self.scale = np.maximum(np.abs(self.jlo), np.abs(self.jhi))
        self.prev_action = np.zeros(len(MOTORS))
        self._stable = 0

    def _obs(self):
        robot = self.agent.robot
        q = np.radians([robot.motor_positions[m] for m in MOTORS])
        dq = np.radians([robot.motor_speeds[m] for m in MOTORS])
        qn = 2 * (q - self.jlo) / (self.jhi - self.jlo) - 1
        # 躯干重力投影：把世界 -z 旋到躯干坐标系
        rot = R.from_quat(robot.global_orientation_quat)  # [x,y,z,w]
        proj_g = rot.inv().apply([0.0, 0.0, -1.0])
        ang_vel = np.radians(robot.gyroscope)  # 陀螺仪已是躯干系 (deg/s)
        return np.concatenate([qn, dq * 0.1, proj_g, ang_vel * 0.25,
                               self.prev_action]).astype(np.float32)

    def execute(self, reset, *args, **kwargs) -> bool:
        if reset:
            self.prev_action = np.zeros(len(MOTORS))
            self._stable = 0

        action = run_network(obs=self._obs(), model=self.model)
        action = np.clip(action, -1, 1)
        self.prev_action = action

        # 动作 -> 目标角（度），按 KP/KD 下发（映射须与 getup_env 一致）
        target_rad = np.clip(action * self.scale, self.jlo, self.jhi)
        for i, m in enumerate(MOTORS):
            self.agent.robot.set_motor_target_position(m, float(np.degrees(target_rad[i])), kp=KP, kd=KD)

        # 完成判定：直立 + 站高，持续若干周期
        rot = R.from_quat(self.agent.robot.global_orientation_quat)
        upright = -rot.inv().apply([0.0, 0.0, -1.0])[2]
        height = self.agent.world.global_position[2]
        if upright > self.UPRIGHT_DONE and height > self.HEIGHT_DONE:
            self._stable += 1
        else:
            self._stable = 0
        return self._stable >= self.STABLE_CYCLES

    def is_ready(self, *args) -> bool:
        return self.agent.world.is_fallen()
