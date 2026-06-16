"""T1 起身强化学习环境（与 rcssservermj 同构）。

物理：服务器自带的 T1 MuJoCo 模型 + 地面，步长 0.005s。
控制：复刻服务器 PD —— torque = kp*(target_rad - q) - kd*dq，按 actuatorfrcrange 限幅。
观测(75) = 关节角(23) + 关节速*0.1(23) + 躯干重力投影(3) + 躯干角速度*0.25(3) + 上一步动作(23)
动作(23) = 每关节目标角，action=0->0弧度(站立)、±1->触及限位。

核心思路（参考 HoST 2025）：训练初期给躯干一个**向上助力**帮它够到站立、拿到站立
奖励，再随训练**退火到 0**，破解"站立动态不稳、RL 探索不到"的死结（如同扶着婴儿学站）。
发现期几乎不加平滑惩罚（参考 HumanUP：先发现能起来的动作，再谈平滑）。
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
    """定位 T1 模型 robot.xml（环境变量 > import rcsssmj > pipx 路径 glob）。"""
    env_path = os.environ.get("T1_ROBOT_XML")
    if env_path and os.path.exists(env_path):
        return env_path
    try:
        import rcsssmj
        p = os.path.join(os.path.dirname(rcsssmj.__file__), "resources", "robots", "T1", "robot.xml")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    import glob
    for pat in [
        os.path.expanduser("~/.local/share/pipx/venvs/*/lib/python*/site-packages/rcsssmj/resources/robots/T1/robot.xml"),
        os.path.expanduser("~/.local/lib/python*/site-packages/rcsssmj/resources/robots/T1/robot.xml"),
        "/usr/lib/python*/site-packages/rcsssmj/resources/robots/T1/robot.xml",
    ]:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    raise FileNotFoundError("找不到 T1 robot.xml；请设 T1_ROBOT_XML 或安装 rcsssmj。")


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
CTRL_DT = 0.02           # 控制周期（s）=> 每动作步进 4 个 0.005 仿真步
EP_TIME = 8.0            # 每回合时长（s）
# 垂直助力课程（弹性支撑式）：向上力 = scale * min(K*(站立高-当前高), FMAX)，
# 越低支撑越大、到站立高归零 => 在"站立"处形成稳定吸引子；scale 随训练退火到 0。
ASSIST_K = 900.0         # 支撑刚度（N/m）
ASSIST_FMAX = 320.0      # 支撑力上限（N，约体重）
ASSIST_ANNEAL = 700_000  # 每环境步数退火到 0（16 env ≈ 全局 1100 万步）


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
        self.total = 0          # 本环境累计步数（用于助力退火）
        self.assist_off = bool(os.environ.get("GETUP_ASSIST_OFF"))  # 续训精修时关助力
        if _HAS_GYM:
            self.action_space = spaces.Box(-1.0, 1.0, (N,), np.float32)
            hi = np.full(75, np.inf, np.float32)
            self.observation_space = spaces.Box(-hi, hi, (75,), np.float32)

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
        self.scale = np.maximum(np.abs(self.jlo), np.abs(self.jhi))
        self.torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    # ---------- 物理/控制 ----------
    def _apply_pd(self, target_rad, assist=0.0):
        q = self.data.qpos[self.qadr]; dq = self.data.qvel[self.dadr]
        tau = np.clip(KP * (target_rad - q) - KD * dq, self.frclo, self.frchi)
        self.data.qfrc_applied[self.dadr] = tau
        self.data.qfrc_applied[2] = assist   # 根 freejoint 的 z 平移自由度 = 世界向上力

    def _action_to_target(self, a):
        a = np.clip(a, -1, 1)
        return np.clip(a * self.scale, self.jlo, self.jhi)

    def _proj_gravity(self):
        R = self.data.xmat[self.torso_bid].reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0])

    def _ang_vel_torso(self):
        R = self.data.xmat[self.torso_bid].reshape(3, 3)
        return R.T @ self.data.qvel[3:6]

    def _obs(self):
        q = self.data.qpos[self.qadr]; dq = self.data.qvel[self.dadr]
        qn = 2 * (q - self.jlo) / (self.jhi - self.jlo) - 1
        return np.concatenate([qn, dq * 0.1, self._proj_gravity(), self._ang_vel_torso() * 0.25,
                               self.prev_action]).astype(np.float32)

    # ---------- reset ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        force_fallen = bool(options and options.get("fallen"))
        if (not force_fallen) and self.rng.random() < 0.5:
            # 参考态：直立躯干 + 随机蹲深，均匀覆盖站立↔深蹲
            c = float(self.rng.random())
            crouch = np.zeros(N)
            for m, v in (("lle1", -1.0), ("rle1", -1.0), ("lle4", 1.6), ("rle4", 1.6),
                         ("lle5", -0.6), ("rle5", -0.6)):
                crouch[MOTORS.index(m)] = v * c
            crouch = np.clip(crouch, self.jlo, self.jhi)
            self.data.qpos[2] = 0.62 - 0.30 * c
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qpos[self.qadr] = crouch + self.rng.uniform(-0.05, 0.05, N)
            mujoco.mj_forward(self.model, self.data)
            for _ in range(60):
                self._apply_pd(crouch); mujoco.mj_step(self.model, self.data)
        else:
            mode = self.rng.integers(0, 3)
            s = {0: [0.7071, 0, -0.7071, 0], 1: [0.7071, 0, 0.7071, 0], 2: [0.7071, 0.7071, 0, 0]}[int(mode)]
            self.data.qpos[2] = 0.35
            self.data.qpos[3:7] = s
            self.data.qpos[self.qadr] = self.rng.uniform(self.jlo, self.jhi) * 0.1
            mujoco.mj_forward(self.model, self.data)
            for _ in range(120):
                self._apply_pd(np.zeros(N)); mujoco.mj_step(self.model, self.data)
        self.prev_action = np.zeros(N)
        self.t = 0
        return self._obs(), {}

    # ---------- step ----------
    def step(self, action):
        action = np.asarray(action, np.float32)
        target = self._action_to_target(action)
        # 弹性支撑助力（随训练退火）：越低于站立高度支撑越大、到站立高归零
        assist_scale = 0.0 if self.assist_off else max(0.0, 1.0 - self.total / ASSIST_ANNEAL)
        for _ in range(self.n_sub):
            h_now = float(self.data.qpos[2])
            support = min(ASSIST_K * max(0.0, STAND_HEIGHT - h_now), ASSIST_FMAX)
            self._apply_pd(target, assist=assist_scale * support)
            mujoco.mj_step(self.model, self.data)
        assist = assist_scale * min(ASSIST_K * max(0.0, STAND_HEIGHT - float(self.data.qpos[2])), ASSIST_FMAX)
        self.prev_action = action
        self.t += 1
        self.total += 1

        h = float(self.data.qpos[2])
        pg = self._proj_gravity()
        upright = float(-pg[2])
        h_frac = min(h / STAND_HEIGHT, 1.0)
        upright01 = max(0.0, upright)
        r_posture = 1.5 * upright01 * h_frac
        # 精修：站立 ramp 门槛抬到 0.50~0.60m，逼它站到接近完全直立才拿满分（更稳更高）
        ramp = float(np.clip((h - 0.50) / (0.60 - 0.50), 0.0, 1.0))
        r_stand = 6.0 * ramp * upright01
        # 站稳加成：高且直立时额外奖励，鼓励稳定保持而非站起即倒
        r_stable = 2.0 if (h > 0.58 and upright > 0.93) else 0.0
        # 精修期稍增平滑惩罚，让起身动作更干净（不抖）
        r_smooth = -0.0002 * float(np.sum(self.data.qvel[self.dadr] ** 2))
        reward = r_posture + r_stand + r_stable + r_smooth + 0.05

        terminated = False
        truncated = self.t >= self.max_steps
        info = {"h": h, "upright": upright, "assist": assist}
        return self._obs(), float(reward), terminated, truncated, info


if __name__ == "__main__":
    env = GetUpEnv()
    obs, _ = env.reset()
    print("观测维度:", obs.shape, " 动作:", N, " 步/回合:", env.max_steps, " 支撑上限:", round(ASSIST_FMAX, 0), "N")
    tot = 0.0
    for i in range(env.max_steps):
        obs, r, te, tr, info = env.step(env.rng.uniform(-1, 1, N)); tot += r
        if i % 80 == 0:
            print(f"  step {i:3d} 高={info['h']:.3f} 直立={info['upright']:+.2f} 助力={info['assist']:.0f}N r={r:.2f}")
        if te or tr: break
    print(f"随机回合总回报={tot:.1f}")
