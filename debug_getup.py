"""
起身动作调试工具。

两个模式，专为"逐帧校验关节方向、微调起身关键帧"设计：

1. 播放模式（推荐先用）：--play back | front
   不管机器人倒没倒，直接循环播放指定的起身关键帧，并在终端实时打印
   「当前第几帧 / 躯干高度 / 躯干姿态(欧拉角)」。配合监视器，你能清楚看到
   每一帧每个关节往哪个方向动 —— 哪个关节动反了，就去对应 yaml 把该帧那个
   值取反。一轮播完会自动重播，方便反复观察。

2. 观察模式（默认，不带 --play）：正常跑球员决策，当机器人摔倒触发起身时，
   打印「选了 front 还是 back / 起身到第几帧 / 躯干高度 / 加速度计」，
   你在监视器里把它推倒即可观察真实触发流程。

用法：
  # 播放仰卧起身，反复观察每帧
  python3 debug_getup.py --play back
  # 播放俯卧起身
  python3 debug_getup.py --play front
  # 观察模式（把球员推倒看真实起身）
  python3 debug_getup.py

可选参数：--host --port -n -t -f（同 run_player.py，-f 默认 7v7）
"""

import argparse
import logging
import os

from mujococodebase.agent import Agent
from mujococodebase.skills.keyframe.keyframe import KeyframeSkill
from mujococodebase.world.play_mode import PlayModeEnum

# 只打印关键信息，关掉框架的 INFO 噪声
logging.basicConfig(level=logging.WARNING)

parser = argparse.ArgumentParser(description="起身动作调试工具")
parser.add_argument("-t", "--team", type=str, default="GetUpDebug", help="队名")
parser.add_argument("-n", "--number", type=int, default=1, help="球员号")
parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器地址")
parser.add_argument("--port", type=int, default=60000, help="服务器端口")
parser.add_argument("-f", "--field", type=str, default="7v7", help="场地（默认 7v7=比赛场地）")
parser.add_argument(
    "--play",
    type=str,
    choices=["back", "front"],
    default=None,
    help="循环播放指定起身关键帧（back=仰卧 / front=俯卧）；不填则为观察模式",
)
args = parser.parse_args()

player = Agent(
    team_name=args.team, number=args.number, host=args.host, port=args.port, field=args.field
)

player.server.connect()
player.server.send_immediate(f"(init {player.robot.name} {player.world.team_name} {player.world.number})")

world = player.world
robot = player.robot

# 播放模式：直接加载对应起身关键帧
keyframe = None
if args.play:
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "mujococodebase", "skills", "keyframe", "get_up",
        f"get_up_{args.play}.yaml",
    )
    keyframe = KeyframeSkill(agent=player, file=yaml_path)
    print(f"[播放模式] 循环播放 get_up_{args.play}.yaml（共 {len(keyframe.keyframes)} 帧）")
    print("配合监视器观察每帧关节方向；动反了就去 yaml 把该帧该值取反。\n")

reset = True
last_print_time = -999.0
last_step = -1


def euler_str():
    """躯干姿态欧拉角(roll, pitch, yaw)，整数度。"""
    e = robot.global_orientation_euler
    return f"r={e[0]:+.0f} p={e[1]:+.0f} y={e[2]:+.0f}"


while True:
    try:
        player.server.receive()
        world.update()

        if args.play:
            # 直接循环播放起身关键帧
            finished = keyframe.execute(reset)
            reset = False
            robot.commit_motor_targets_pd()

            step = keyframe.keyframe_step
            if step != last_step:  # 只在切帧时打印，清爽
                print(
                    f"t={world.server_time:6.1f}  帧 {step}/{len(keyframe.keyframes)}"
                    f"  躯干高度={world.global_position[2]:.2f}m  姿态[{euler_str()}]"
                )
                last_step = step

            if finished:
                print("--- 一轮播放结束，0.0 重播 ---\n")
                reset = True
                last_step = -1
        else:
            # 观察模式：正常决策，摔倒触发起身时打印
            player.decision_maker.update_current_behavior()

            t = world.server_time or 0.0
            if t - last_print_time >= 0.3:  # 限频
                last_print_time = t
                gu = player.skills_manager.skills.get("GetUp")
                chosen = getattr(gu, "chosen_get_up", None)
                variant = "—"
                step = "—"
                if chosen is gu.get_up_front:
                    variant, step = "front", chosen.keyframe_step
                elif chosen is gu.get_up_back:
                    variant, step = "back", chosen.keyframe_step
                print(
                    f"t={t:6.1f}  pm={world.playmode.name:14s}  "
                    f"高度={world.global_position[2]:.2f}m  倒地={world.is_fallen()}  "
                    f"accel[0]={robot.accelerometer[0]:+.1f}  起身={variant} 帧{step}  姿态[{euler_str()}]"
                )

        player.server.send()
    except Exception:
        player.shutdown()
        raise
