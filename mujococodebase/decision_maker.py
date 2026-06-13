from dataclasses import Field
import logging
from typing import Mapping

import numpy as np
from mujococodebase.utils.math_ops import MathOps
from mujococodebase.world.field import FIFAField, HLAdultField, SevenVSevenField
from mujococodebase.world.play_mode import PlayModeEnum, PlayModeGroupEnum


logger = logging.getLogger()


class DecisionMaker:
    """
    Responsible for deciding what the agent should do at each moment.

    This class is called every simulation step to update the agent's behavior
    based on the current state of the world and game conditions.
    """

    BEAM_POSES: Mapping[type[Field], Mapping[int, tuple[float, float, float]]] ={
        FIFAField: {
            1: (2.1, 0, 0),
            2: (22.0, 12.0, 0),
            3: (22.0, 4.0, 0),
            4: (22.0, -4.0, 0),
            5: (22.0, -12.0, 0),
            6: (15.0, 0.0, 0),
            7: (4.0, 16.0, 0),
            8: (11.0, 6.0, 0),
            9: (11.0, -6.0, 0),
            10: (4.0, -16.0, 0),
            11: (7.0, 0.0, 0),
        },
        HLAdultField: {
            1: (7.0, 0.0, 0),
            2: (2.0, -1.5, 0),
            3: (2.0, 1.5, 0),
        },
        SevenVSevenField: {
            1: (-25.0, 0.0, 0),
            2: (-18.0, -8.0, 0),
            3: (-18.0, 8.0, 0),
            4: (-5.0, -10.0, 0),
            5: (-10.0, 0.0, 0),
            6: (-5.0, 10.0, 0),
            7: (10.0, 0.0, 0),
        },
    }

    def __init__(self, agent):
        """
        Creates a new DecisionMaker linked to the given agent.

        Args:
            agent: The main agent that owns this DecisionMaker.
        """
        from mujococodebase.agent import Agent  # type hinting

        self.agent: Agent = agent
        self.is_getting_up: bool = False

    def update_current_behavior(self) -> None:
        """
        决定 agent 在当前这一步应该做什么。

        每个仿真周期都会调用本函数：先处理通用情况（比赛结束、开球前/进球后
        的传送站位、倒地起身），随后根据"球员角色 + 当前比赛模式"分发到具体行为。
        """

        world = self.agent.world

        # 比赛结束，不再下发任何动作
        if world.playmode is PlayModeEnum.GAME_OVER:
            return

        # 开球前 / 进球后：把球员传送回各自的初始站位
        if world.playmode_group in (
            PlayModeGroupEnum.ACTIVE_BEAM,
            PlayModeGroupEnum.PASSIVE_BEAM,
        ):
            self.agent.server.commit_beam(
                pos2d=self.BEAM_POSES[type(world.field)][world.number][:2],
                rotation=self.BEAM_POSES[type(world.field)][world.number][2],
            )

        # 倒地起身拥有最高优先级：只要正在起身或满足起身条件，就先把人扶起来
        if self.is_getting_up or self.agent.skills_manager.is_ready(skill_name="GetUp"):
            self.is_getting_up = not self.agent.skills_manager.execute(skill_name="GetUp")
            self.agent.robot.commit_motor_targets_pd()
            return

        playmode = world.playmode
        group = world.playmode_group

        if playmode in (
            PlayModeEnum.BEFORE_KICK_OFF,
            PlayModeEnum.OUR_GOAL,
            PlayModeEnum.THEIR_GOAL,
        ):
            # 开球前与进球后：在站位上保持中性站立姿态
            self.agent.skills_manager.execute("Neutral")

        elif self.is_goalkeeper():
            # 守门员：无论何种模式都执行守门行为（不参与抢球/带球）
            self.goalkeeper_behavior()

        elif group is PlayModeGroupEnum.THEIR_KICK:
            # 对方死球（对方开球 / 任意球 / 角球 / 界外球 / 球门球等）：
            # 严禁抢先碰球，全员回防站位并与球保持安全距离
            self.defensive_positioning()

        elif playmode is PlayModeEnum.PLAY_ON or group is PlayModeGroupEnum.OUR_KICK:
            # 比赛进行中，或我方死球（由我方发球）：
            # 只有"阵型上离球最近"的球员去处理球，其余球员回到各自站位
            if self.should_chase_ball():
                self.carry_ball()
            else:
                self.move_to_formation()

        else:
            # 兜底：保持阵型站位
            self.move_to_formation()

        self.agent.robot.commit_motor_targets_pd()

    # ----------------------------- 角色与站位辅助 -----------------------------

    def is_goalkeeper(self) -> bool:
        """1 号球员为守门员（规则要求守门员球衣号必须为 1 号）。"""
        return self.agent.world.number == 1

    def home_position(self, number: int | None = None) -> np.ndarray:
        """返回某号球员在当前场地下的初始站位（阵型基准点，2D）。"""
        if number is None:
            number = self.agent.world.number
        return np.array(self.BEAM_POSES[type(self.agent.world.field)][number][:2])

    def should_chase_ball(self) -> bool:
        """
        判断"是否轮到我去处理球"。

        由于队友的实时位置只在视野内才可见、并不可靠，这里改用所有球员共享的
        固定阵型站位来判断：在全部外场球员中，谁的初始站位离球最近，就由谁追球。
        每个 agent 都用相同的（阵型 + 全局球位）信息独立计算，结果一致，无需通信。
        """
        world = self.agent.world
        ball = world.ball_pos[:2]
        poses = self.BEAM_POSES[type(world.field)]
        outfielders = [n for n in poses if n != 1]  # 排除守门员
        closest = min(
            outfielders,
            key=lambda n: np.linalg.norm(np.array(poses[n][:2]) - ball),
        )
        return world.number == closest

    def move_to_formation(self) -> None:
        """回到自己的阵型站位；到位后保持中性站立，并大致面向球。"""
        world = self.agent.world
        my_pos = world.global_position[:2]
        home = self.home_position()
        ball = world.ball_pos[:2]

        dist_to_home = np.linalg.norm(my_pos - home)
        if dist_to_home < 0.4:
            self.agent.skills_manager.execute("Neutral")
            return

        desired_orientation = MathOps.vector_angle(ball - my_pos)
        self.agent.skills_manager.execute(
            "Walk",
            target_2d=home,
            is_target_absolute=True,
            orientation=desired_orientation if dist_to_home <= 2 else None,
        )

    def defensive_positioning(self) -> None:
        """
        对方死球时的防守站位：回到阵型站位，但绝不主动碰球。

        若站位点或自身当前位置距离球过近（小于安全距离），则沿"远离球"的方向
        后撤到安全距离之外，避免抢先触球造成违规。
        """
        SAFE_DISTANCE = 2.5  # 与球应保持的安全距离（米）

        world = self.agent.world
        my_pos = world.global_position[:2]
        ball = world.ball_pos[:2]
        target = self.home_position().astype(float)

        # 站位点离球太近：把目标点沿"球→站位点"方向推到安全距离外
        if np.linalg.norm(target - ball) < SAFE_DISTANCE:
            away = target - ball
            norm = np.linalg.norm(away)
            if norm < 1e-6:
                # 退化情况：朝我方球门方向后撤
                our_goal = np.array(world.field.get_our_goal_position(), dtype=float)
                away = our_goal - ball
                norm = np.linalg.norm(away) + 1e-9
            target = ball + away / norm * SAFE_DISTANCE

        # 我当前就站得离球太近：直接朝远离球的方向后撤
        if np.linalg.norm(my_pos - ball) < SAFE_DISTANCE:
            away = my_pos - ball
            norm = np.linalg.norm(away)
            away = away / norm if norm > 1e-6 else np.array([-1.0, 0.0])
            target = ball + away * SAFE_DISTANCE

        dist_to_target = np.linalg.norm(my_pos - target)
        if dist_to_target < 0.4:
            self.agent.skills_manager.execute("Neutral")
            return

        desired_orientation = MathOps.vector_angle(ball - my_pos)
        self.agent.skills_manager.execute(
            "Walk",
            target_2d=target,
            is_target_absolute=True,
            orientation=desired_orientation if dist_to_target <= 2 else None,
        )

    def goalkeeper_behavior(self) -> None:
        """
        守门员行为。

        默认在球门线前做横向拦截站位：x 固定在门前一点，y 跟随球（限制在门柱
        范围内），并始终面向球。仅当比赛进行（PLAY_ON）且球进入本方门前的危险
        区域时，才出击上前解围（把球向前带离危险区）。其余模式（尤其对方死球）
        一律保持站位，绝不冲出去碰球。
        """
        world = self.agent.world
        my_pos = world.global_position[:2]
        ball = world.ball_pos[:2]

        field_length = world.field.get_length()
        field_width = world.field.get_width()
        goal_x = world.field.get_our_goal_position()[0]  # 我方球门 x（位于 -L/2）

        GOAL_LINE_OFFSET = 2.0                  # 站在球门线前方的距离（米）
        goal_half_width = 0.045 * field_width   # 横向移动半幅，按场地宽度自适应

        # 危险区：球进入本方半场靠近球门的区域
        danger_x = goal_x + 0.20 * field_length
        ball_in_danger = (ball[0] < danger_x) and (abs(ball[1]) < 0.30 * field_width)

        # 仅在比赛进行且球进入危险区时出击解围
        if world.playmode is PlayModeEnum.PLAY_ON and ball_in_danger:
            self.carry_ball()
            return

        # 横向拦截站位
        target = np.array([
            goal_x + GOAL_LINE_OFFSET,
            np.clip(ball[1], -goal_half_width, goal_half_width),
        ])

        dist_to_target = np.linalg.norm(my_pos - target)
        desired_orientation = MathOps.vector_angle(ball - my_pos)
        if dist_to_target < 0.3:
            self.agent.skills_manager.execute("Neutral")
            return

        self.agent.skills_manager.execute(
            "Walk",
            target_2d=target,
            is_target_absolute=True,
            orientation=desired_orientation,
        )

    def carry_ball(self):
        """
        Basic example of a behavior: moves the robot toward the goal while handling the ball.
        """
        their_goal_pos = self.agent.world.field.get_their_goal_position()[:2]
        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]

        ball_to_goal = their_goal_pos - ball_pos
        bg_norm = np.linalg.norm(ball_to_goal)
        if bg_norm == 0:
            return 
        ball_to_goal_dir = ball_to_goal / bg_norm

        dist_from_ball_to_start_carrying = 0.30
        carry_ball_pos = ball_pos - ball_to_goal_dir * dist_from_ball_to_start_carrying

        my_to_ball = ball_pos - my_pos
        my_to_ball_norm = np.linalg.norm(my_to_ball)
        if my_to_ball_norm == 0:
            my_to_ball_dir = np.zeros(2)
        else:
            my_to_ball_dir = my_to_ball / my_to_ball_norm

        cosang = np.dot(my_to_ball_dir, ball_to_goal_dir)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle_diff = np.arccos(cosang)

        ANGLE_TOL = np.deg2rad(7.5)
        aligned = (my_to_ball_norm > 1e-6) and (angle_diff <= ANGLE_TOL)

        behind_ball = np.dot(my_pos - ball_pos, ball_to_goal_dir) < 0
        desired_orientation = MathOps.vector_angle(ball_to_goal)

        if not aligned or not behind_ball:
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=carry_ball_pos,
                is_target_absolute=True,
                orientation=None if np.linalg.norm(my_pos - carry_ball_pos) > 2 else desired_orientation
            )
        else:
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=their_goal_pos,
                is_target_absolute=True,
                orientation=desired_orientation
            )

