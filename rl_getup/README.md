# RL 起身策略训练

用强化学习训练 T1 的起身动作，替代脆弱的关键帧起身。
背景：开环关键帧没有平衡反馈，连"蹲下再站起"都会倒（见离线仿真验证），
所以起身必须用带实时反馈的策略——和走路 `walk.onnx` 同思路。**自己训练 = 自己的成果。**

## 原理
- **环境** [getup_env.py](getup_env.py)：服务器自带的 T1 MuJoCo 模型 + 地面，复刻服务器
  PD 控制律（`torque = kp*(target-q) - kd*dq`，力矩按关节 `actuatorfrcrange` 限幅，步长 0.005s）。
  机器人从随机倒地姿态开始，奖励 = 直立 + 升高 + 站立bonus − 能耗/抖动。
- **观测/动作刻意只用部署时 agent 真能拿到/下发的量**，保证策略可直接接回比赛：
  - 观测(75) = 关节角(23)+关节速(23)+躯干重力投影(3)+躯干角速度(3)+上一步动作(23)
  - 动作(23) = 每关节目标角（[-1,1] 映射到限位）
- 训练产物导成 `get_up.onnx`，由 [GetUpRL 技能](../mujococodebase/skills/rl/get_up_rl.py) 用 onnxruntime 加载，
  观测构造/动作映射/增益与训练环境**逐位对齐**。

## 步骤

```bash
# 1. 装训练依赖（独立 venv，torch 较大）
bash rl_getup/setup.sh

# 2. 先小步数验证奖励是否上升（冒烟测试）
rl_getup/.venv_rl/bin/python rl_getup/train.py --steps 200000 --n-envs 8

# 3. 正式训练（几百万~上千万步；无 GPU 用 CPU 多环境，耗时较长，可后台跑）
rl_getup/.venv_rl/bin/python rl_getup/train.py --steps 8000000 --n-envs 16

# 4. 导出 onnx 到技能目录
rl_getup/.venv_rl/bin/python rl_getup/export_onnx.py
# -> mujococodebase/skills/rl/get_up.onnx
```

## 接入比赛 agent
训练并导出 `get_up.onnx` 后，在 [skills_manager.py](../mujococodebase/skills/skills_manager.py)
的 `create_skills` 里把 `GetUpRL` 注册进来，并在 [decision_maker.py](../mujococodebase/decision_maker.py)
的起身分支用 `GetUpRL` 替换关键帧版 `GetUp`（两者接口一致：`execute(reset)->bool` / `is_ready()`）。
未导出 onnx 前不要注册，否则会因找不到模型报错。

## 离线验证
没有 onnx 也能用 mujoco 跑环境自检（确认物理/奖励正常）：
```bash
~/.local/share/pipx/venvs/rcsssmj/bin/python rl_getup/getup_env.py
```

## 调参提示
- 起身学不出来时：先确认随机策略回合回报为负、且训练中 `ep_rew_mean` 持续上升。
- 站起但抖/不稳：加大 `r_smooth` 惩罚或延长回合 `EP_TIME`。
- 只会坐不会站：提高 `r_stand` 权重与触发高度，或课程式先训直立再训起身。
- `KP/KD/CTRL_DT` 改了，必须同步改 GetUpRL 技能里的同名常量。
