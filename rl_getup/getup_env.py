"""T1 起身强化学习环境（与 rcssservermj 同构）。

物理：服务器自带的 T1 MuJoCo 模型 + 地面，步长 0.005s。
控制：复刻服务器 PD —— torque = kp*(target_rad - q) - kd*dq，按 actuatorfrcrange 限幅。
观测/动作：刻意只用「部署时 agent 真能拿到/下发」的量，保证训练出的策略可直接接回比赛：
  - 观测(75)= 关节角(23) + 关节速(23) + 躯干重力投影(3) + 躯干角速度(3) + 上一步动作(23)
  - 动作(23)= 每个关节的目标角，[-1,1] 线性映射到该关节限位
起身奖励：直立 + 升高 + 站立bonus − 能耗/抖动。

核心 reset/step 只依赖 mujoco+numpy（可单独测试）；若装了 gymnasium 则暴露标准 Env 接口。
"""
import os
import numpy as np
import mujoco

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
    _Base = gym.Env
except Exception:
    _HAS_GYM = False
    _Base = object


def _robot_xml():
    """定位 T1 模型 robot.xml。

    优先用环境变量 T1_ROBOT_XML；否则尝试 import rcsssmj（若该 venv 装了）；
    最后回退到已知的 pipx 安装路径 glob。这样训练 venv 不必安装 rcsssmj。
    """
    # 1) 环境变量显式指定
    env_path = os.environ.get("T1_ROBOT_XML")
    if env_path and os.path.exists(env_path):
        return env_path
    # 2) 若当前 venv 恰好装了 rcsssmj
    try:
        import rcsssmj
        p = os.path.join(os.path.dirname(rcsssmj.__file__), "resources", "robots", "T1", "robot.xml")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    # 3) 回退：在 pipx/常见 site-packages 里搜
    import glob
    patterns = [
        os.path.expanduser("~/.local/share/pipx/venvs/*/lib/python*/site-packages/rcsssmj/resources/robots/T1/robot.xml"),
        os.path.expanduser("~/.local/lib/python*/site-packages/rcsssmj/resources/robots/T1/robot.xml"),
        "/usr/lib/python*/site-packages/rcsssmj/resources/robots/T1/robot.xml",
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "找不到 T1 robot.xml。请设置环境变量 T1_ROBOT_XML 指向 rcsssmj 的 "
        "resources/robots/T1/robot.xml，或在本 venv 安装 rcsssmj。"
    )


# 服务器电机码顺序（动作/观测的关节顺序，与 robot.py ROBOT_MOTORS 一致）
MOTORS = ["he1","he2","lae1","lae2","lae3","lae4","rae1","rae2","rae3","rae4","te1",
          "lle1","lle2","lle3","lle4","lle5","lle6","rle1","rle2","rle3","rle4","rle5","rle6"]
MOTOR2JOINT = {
    "he1":"AAHead_yaw","he2":"Head_pitch","lae1":"Left_Shoulder_Pitch","lae2":"Left_Shoulder_Roll",
    "lae3":"Left_Elbow_Pitch","lae4":"Left_Elbow_Yaw","rae1":"Right_Shoulder_Pitch","rae2":"Right_Shoulder_Roll",
    "rae3":"Right_Elbow_Pitch","rae4":"Right_Elbow_Yaw","te1":"Waist",
    "lle1":"Left_Hip_Pitch","lle2":"Left_Hip_Roll","lle3":"Left_Hip_Yaw","lle4":"Left_Knee_Pitch","lle5":"Left_Ankle_Pitch","lle6":"Left_Ankle_Roll",
    "rle1":"Right_Hip_Pitch","rle2":"Right_Hip_Roll","rle3":"Right_Hip_Yaw","rle4":"Right_Knee_Pitch","rle5":"Right_Ankle_Pitch","rle6":"Right_Ankle_Roll",
}

N = len(MOTORS)
STAND_HEIGHT = 0.62      # 站立目标躯干高度（m）
KP, KD = 200.0, 5.0      # PD 增益（部署的 GetUpRL 技能须用同值）
CTRL_DT = 0.02           # 控制周期（s）=> 每个动作步进 4 个 0.005 仿真步
EP_TIME = 6.0            # 每回合时长（s）


class GetUpEnv(_Base):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        super().__init__()
        self.model = self._build()
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self._index()
        self.n_sub = int(round(CTRL_DT / self.model.opt.timestep))
        self.max_steps = int(EP_TIME / CTRL_DT)
        self.prev_action = np.zeros(N)
        self.t = 0
        if _HAS_GYM:
            self.action_space = spaces.Box(-1.0, 1.0, (N,), np.float32)
            hi = np.full(75, np.inf, np.float32)
            self.observation_space = spaces.Box(-hi, hi, (75,), np.float32)

    # ---------- 建模 ----------
    def _build(self):
        spec = mujoco.MjSpec.from_file(_robot_xml())
        spec.option.timestep = 0.005
        g = spec.worldbody.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_PLANE
        g.size = [0, 0, 1]; g.pos = [0, 0, 0]; g.friction = [1.0, 0.01, 0.005]
        return spec.compile()

    def _index(self):
        self.qadr = np.zeros(N, int); self.dadr = np.zeros(N, int)
        self.jlo = np.zeros(N); self.jhi = np.zeros(N)
        self.frclo = np.zeros(N); self.frchi = np.zeros(N)
        for i, mo in enumerate(MOTORS):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, MOTOR2JOINT[mo])
            self.qadr[i] = self.model.jnt_qposadr[jid]
            self.dadr[i] = self.model.jnt_dofadr[jid]
            self.jlo[i], self.jhi[i] = self.model.jnt_range[jid]
            if self.model.jnt_actfrclimited[jid]:
                self.frclo[i], self.frchi[i] = self.model.jnt_actfrcrange[jid]
            else:
                self.frclo[i], self.frchi[i] = -1e6, 1e6
        # 动作幅度：每关节取 max(|lo|,|hi|)，使 action=0->0 弧度(站立)、±1 触及限位
        self.scale = np.maximum(np.abs(self.jlo), np.abs(self.jhi))
        self.torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # ---------- 物理/控制 ----------
    def _apply_pd(self, target_rad):
        q = self.data.qpos[self.qadr]; dq = self.data.qvel[self.dadr]
        tau = KP * (target_rad - q) - KD * dq
        tau = np.clip(tau, self.frclo, self.frchi)
        self.data.qfrc_applied[self.dadr] = tau

    def _action_to_target(self, a):
        # action=0 -> 关节 0 弧度（站立姿态），±1 -> 触及该关节较远的限位。
        # 这样"站立"是默认动作，蹲下需主动出力，避免动作参数化把策略带向蹲姿。
        a = np.clip(a, -1, 1)
        return np.clip(a * self.scale, self.jlo, self.jhi)

    # ---------- 观测 ----------
    def _proj_gravity(self):
        R = self.data.xmat[self.torso_bid].reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0])

    def _ang_vel_torso(self):
        R = self.data.xmat[self.torso_bid].reshape(3, 3)
        return R.T @ self.data.qvel[3:6]

    def _obs(self):
        q = self.data.qpos[self.qadr]
        dq = self.data.qvel[self.dadr]
        qn = 2 * (q - self.jlo) / (self.jhi - self.jlo) - 1  # 归一化到 [-1,1]
        return np.concatenate([qn, dq * 0.1, self._proj_gravity(), self._ang_vel_torso() * 0.25,
                               self.prev_action]).astype(np.float32)

    # ---------- reset ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.rng.random() < 0.5:
            # 参考态初始化：直立躯干 + 随机蹲深（c=0 站直 ~ c=1 深蹲），
            # 覆盖「站立↔深蹲」整个高度段，密集训练"蹲→站"这一跃；
            # 价值函数再把「站起=高分且可达」反传到倒地状态。
            c = float(self.rng.random())
            crouch = np.zeros(N)
            for m, v in (("lle1", -1.0), ("rle1", -1.0), ("lle4", 1.6), ("rle4", 1.6),
                         ("lle5", -0.6), ("rle5", -0.6)):
                crouch[MOTORS.index(m)] = v * c          # 屈髋负/屈膝正/踝背屈负
            crouch = np.clip(crouch, self.jlo, self.jhi)
            self.data.qpos[2] = 0.62 - 0.30 * c
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]            # 躯干直立
            self.data.qpos[self.qadr] = crouch + self.rng.uniform(-0.05, 0.05, N)
            mujoco.mj_forward(self.model, self.data)
            for _ in range(60):  # 保持蹲姿落定
                self._apply_pd(crouch); mujoco.mj_step(self.model, self.data)
        else:
            # 倒地姿态：绕 y 转 ±90°(俯/仰卧) 或侧卧，落地稳定
            mode = self.rng.integers(0, 3)
            s = {0: [0.7071, 0, -0.7071, 0], 1: [0.7071, 0, 0.7071, 0], 2: [0.7071, 0.7071, 0, 0]}[int(mode)]
            self.data.qpos[2] = 0.35
            self.data.qpos[3:7] = s
            self.data.qpos[self.qadr] = self.rng.uniform(self.jlo, self.jhi) * 0.1
            mujoco.mj_forward(self.model, self.data)
            for _ in range(120):  # 0.6s 落地
                self._apply_pd(np.zeros(N)); mujoco.mj_step(self.model, self.data)
        self.prev_action = np.zeros(N)
        self.t = 0
        return self._obs(), {}

    # ---------- step ----------
    def step(self, action):
        action = np.asarray(action, np.float32)
        target = self._action_to_target(action)
        for _ in range(self.n_sub):
            self._apply_pd(target)
            mujoco.mj_step(self.model, self.data)
        self.prev_action = action
        self.t += 1

        h = float(self.data.qpos[2])
        pg = self._proj_gravity()
        upright = float(-pg[2])                 # 直立时 ≈ +1
        h_frac = min(h / STAND_HEIGHT, 1.0)
        upright01 = max(0.0, upright)
        # 关键：直立必须「配合站高」才给分（乘积），杜绝"蹲着保持竖直"的偷懒局部最优
        r_posture = upright01 * h_frac
        r_tall = h_frac                          # 额外直接鼓励站高
        # 站立奖励改为平滑爬坡：在 0.35~0.58m 之间从 0 线性升到 1（须直立），
        # 持续穿过"深蹲高度"往上拉，而非 h>0.5 的硬阈值（之前从未触发）。
        ramp = float(np.clip((h - 0.35) / (0.58 - 0.35), 0.0, 1.0))
        r_stand = ramp * upright01
        r_ctrl = -0.001 * float(np.sum(action ** 2))
        r_smooth = -0.0005 * float(np.sum(self.data.qvel[self.dadr] ** 2))
        reward = 3.0 * r_posture + 1.0 * r_tall + 4.0 * r_stand + r_ctrl + r_smooth + 0.05

        terminated = False
        truncated = self.t >= self.max_steps
        info = {"h": h, "upright": upright, "r_stand": r_stand}
        return self._obs(), float(reward), terminated, truncated, info


# 不依赖 gymnasium 的快速自测
if __name__ == "__main__":
    env = GetUpEnv()
    obs, _ = env.reset()
    print("观测维度:", obs.shape, " 动作维度:", N, " 每回合步数:", env.max_steps, " 子步/动作:", env.n_sub)
    tot = 0.0
    for i in range(env.max_steps):
        obs, r, term, trunc, info = env.step(env.rng.uniform(-1, 1, N))
        tot += r
        if i % 50 == 0:
            print(f"  step {i:3d}  高度={info['h']:.3f}  直立={info['upright']:+.2f}  reward={r:.2f}")
        if term or trunc:
            break
    print(f"随机策略回合总回报={tot:.1f}（应该很低；训练后应显著上升并学会站起）")
