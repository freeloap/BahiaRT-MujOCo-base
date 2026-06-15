"""起身动作离线仿真调试工具（与 rcssservermj 同构）。

直接读取 mujococodebase/skills/keyframe/get_up/*.yaml，在本地 MuJoCo 里从
仰卧/俯卧姿态仿真起身，报告躯干高度轨迹与是否站起。改 yaml → 重跑 → 看结果，
几秒一轮，无需开游戏服务器。

控制律复刻服务器：torque = kp*(target_rad - q) - kd*dq，按各关节 actuatorfrcrange
限幅；步长 0.005s；目标角先按关节限位钳制（与 robot.py 一致）。

⚠️ 必须用服务器的 venv 运行（带 mujoco 与 T1 模型）：
    /home/dumbcat/.local/share/pipx/venvs/rcsssmj/bin/python tools/getup_sim.py back
    加 --view 打开窗口实时观看：... tools/getup_sim.py back --view

参数：
    位置参数 back|front：调哪个起身（决定读哪个 yaml、初始姿态仰/俯卧）
    --view  打开 MuJoCo 可视化窗口
"""
import argparse
import os
import sys

import numpy as np
import yaml
import mujoco

# 服务器电机码 -> 模型关节名
MOTOR2JOINT = {
    "he1": "AAHead_yaw", "he2": "Head_pitch",
    "lae1": "Left_Shoulder_Pitch", "lae2": "Left_Shoulder_Roll", "lae3": "Left_Elbow_Pitch", "lae4": "Left_Elbow_Yaw",
    "rae1": "Right_Shoulder_Pitch", "rae2": "Right_Shoulder_Roll", "rae3": "Right_Elbow_Pitch", "rae4": "Right_Elbow_Yaw",
    "te1": "Waist",
    "lle1": "Left_Hip_Pitch", "lle2": "Left_Hip_Roll", "lle3": "Left_Hip_Yaw", "lle4": "Left_Knee_Pitch", "lle5": "Left_Ankle_Pitch", "lle6": "Left_Ankle_Roll",
    "rle1": "Right_Hip_Pitch", "rle2": "Right_Hip_Roll", "rle3": "Right_Hip_Yaw", "rle4": "Right_Knee_Pitch", "rle5": "Right_Ankle_Pitch", "rle6": "Right_Ankle_Roll",
}

# 可读关节组 -> (server电机名对, 是否镜像)；与 robot.py 的 MOTOR_SYMMETRY 一致
SYMMETRY = {
    "Head_yaw": (("he1",), False), "Head_pitch": (("he2",), False),
    "Shoulder_Pitch": (("lae1", "rae1"), False), "Shoulder_Roll": (("lae2", "rae2"), True),
    "Elbow_Pitch": (("lae3", "rae3"), False), "Elbow_Yaw": (("lae4", "rae4"), True),
    "Waist": (("te1",), False),
    "Hip_Pitch": (("lle1", "rle1"), False), "Hip_Roll": (("lle2", "rle2"), True),
    "Hip_Yaw": (("lle3", "rle3"), True), "Knee_Pitch": (("lle4", "rle4"), False),
    "Ankle_Pitch": (("lle5", "rle5"), False), "Ankle_Roll": (("lle6", "rle6"), True),
}


def robot_xml_path():
    """从已安装的 rcsssmj 包定位 T1 模型。"""
    import rcsssmj
    return os.path.join(os.path.dirname(rcsssmj.__file__), "resources", "robots", "T1", "robot.xml")


def yaml_path(which):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "mujococodebase", "skills", "keyframe", "get_up", f"get_up_{which}.yaml")


def frame_to_motor_targets(motor_positions):
    """把一帧（可读关节->yaml值，symmetry约定）展开成 {电机: 实际目标角(度)}。
    复刻 KeyframeSkill：非镜像组两侧 = -yaml；镜像组 左=+yaml/右=-yaml。"""
    out = {}
    for jn, val in motor_positions.items():
        motors, inverse = SYMMETRY[jn]
        for k, m in enumerate(motors):
            out[m] = (val if (inverse and k == 0) else -val)
    return out


def build():
    spec = mujoco.MjSpec.from_file(robot_xml_path())
    spec.option.timestep = 0.005
    g = spec.worldbody.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [0, 0, 1]; g.pos = [0, 0, 0]; g.friction = [1.0, 0.01, 0.005]
    model = spec.compile()
    return model, mujoco.MjData(model)


def addr(model):
    info = {}
    for m, jn in MOTOR2JOINT.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        info[m] = dict(
            qadr=model.jnt_qposadr[jid], dadr=model.jnt_dofadr[jid],
            frc=(model.jnt_actfrcrange[jid] if model.jnt_actfrclimited[jid] else np.array([-1e6, 1e6])),
            rng=np.degrees(model.jnt_range[jid]),
        )
    return info


def torso_pitch_deg(data):
    w, x, y, z = data.qpos[3:7]
    return float(np.degrees(np.arcsin(np.clip(-2 * (x * z - w * y), -1, 1))))


def control(model, data, info, targets_deg, kp, kd):
    data.qfrc_applied[:] = 0.0
    for m, a in info.items():
        if m not in targets_deg:
            continue
        lo, hi = a["rng"]
        tgt = np.radians(max(lo, min(hi, targets_deg[m])))   # 关节限位钳制
        q = data.qpos[a["qadr"]]; dq = data.qvel[a["dadr"]]
        tau = kp * (tgt - q) - kd * dq
        flo, fhi = a["frc"]
        data.qfrc_applied[a["dadr"]] = max(flo, min(fhi, tau))


def set_fallen(model, data, info, which):
    """设置仰卧(back)/俯卧(front)初始姿态并落地稳定。"""
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.35
    # 绕 y 轴 ±90°：back=仰卧(-90)，front=俯卧(+90)
    s = -0.7071 if which == "back" else 0.7071
    data.qpos[3:7] = [0.7071, 0, s, 0]
    mujoco.mj_forward(model, data)
    for _ in range(120):  # 0.6s 落地稳定，关节保持 0
        control(model, data, info, {m: 0.0 for m in info}, 120, 5)
        mujoco.mj_step(model, data)


def main():
    ap = argparse.ArgumentParser(description="起身动作离线仿真")
    ap.add_argument("which", choices=["back", "front"], help="调哪个起身")
    ap.add_argument("--view", action="store_true", help="打开可视化窗口观看")
    args = ap.parse_args()

    desc = yaml.safe_load(open(yaml_path(args.which)))
    base_kp, base_kd = desc["kp"], desc["kd"]
    frames = desc["keyframes"]

    model, data = build()
    info = addr(model)
    set_fallen(model, data, info, args.which)

    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    print(f"[{args.which}] 初始: 高度={data.qpos[2]:.3f}m 俯仰≈{torso_pitch_deg(data):+.0f}°  共 {len(frames)} 帧")
    maxz = data.qpos[2]
    for i, kf in enumerate(frames):
        tgt = frame_to_motor_targets(kf["motor_positions"])
        kp = kf.get("kp", base_kp); kd = kf.get("kd", base_kd)
        steps = max(1, int(kf["delta"] / 0.005))
        for _ in range(steps):
            control(model, data, info, tgt, kp, kd)
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
        maxz = max(maxz, data.qpos[2])
        print(f"  帧{i} ({kf['delta']}s kp={kp}): 高度={data.qpos[2]:.3f}m 俯仰≈{torso_pitch_deg(data):+.0f}°")

    # 末帧再保持 1.5s 看是否站稳
    last = frame_to_motor_targets(frames[-1]["motor_positions"])
    for _ in range(300):
        control(model, data, info, last, base_kp, base_kd)
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
    zf = data.qpos[2]
    print(f"末态: 高度={zf:.3f}m  峰值={maxz:.3f}m")
    print("结论:", "✅ 站起来了" if zf > 0.5 else ("⚠️ 起来过但没站稳" if maxz > 0.45 else "❌ 没起来"))
    if viewer is not None:
        print("窗口已开，按 Ctrl+C 结束")
        try:
            while viewer.is_running():
                viewer.sync()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
