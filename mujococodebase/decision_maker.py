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
    决定 agent 每一步该做什么。

    每个仿真周期调用一次：先处理通用情况（比赛结束、开球/进球后的传送站位、
    倒地起身），再按「球员角色 + 当前比赛模式」分发到具体行为。

    设计原则（无通信）：队友实时位置只在视野内可见、并不可靠，因此所有「分工」
    都基于全体共享的固定阵型站位独立计算，每个 agent 用相同输入得到一致结果，
    无需互相通信（规则禁止球员进程间私有通信）。

    坐标系：world_parser 已对右队做 180° 翻转，所有 agent 都以「己方球门在 -x、
    进攻朝 +x」的统一视角运作。
    """

    # ----------------------------- 可调参数（互踢实测后微调）-----------------------------
    # 数值按比赛场地 fifa7vs7（55×36m，球门在 x=±27.5）标定。
    NUM_CHASERS: int = 2          # 同时上前追球/带球的外场球员数
    SAFE_DISTANCE: float = 2.5    # 对方死球时与球应保持的安全距离（米）
    FORMATION_SHIFT_K: float = 0.45   # 阵型随球纵向平移的比例
    FORMATION_SHIFT_MAX: float = 8.0  # 阵型纵向平移的最大幅度（米）
    ARRIVE_TOL: float = 0.4       # 外场球员到位判定阈值（米）
    GK_ARRIVE_TOL: float = 0.3    # 守门员到位判定阈值（米）
    ORIENT_NEAR: float = 2.0      # 距目标小于此值时才锁定朝向（米）
    GK_LINE_OFFSET: float = 2.0   # 守门员站在己方端线前方的距离（米）
    GK_HALF_WIDTH: float = 2.5    # 守门员横向跟球的半幅（米，略宽于球门半宽 1.83）
    DRIBBLE_BEHIND: float = 0.30  # 带球时站到球后方的距离（米）
    DRIBBLE_MAX: float = 0.7      # 单次持续带球的最大推进距离（米，< 规则 1.0）
    DRIBBLE_RESET_TIME: float = 0.6   # 带球过头后退开重置持球计时的时长（秒，> 规则 0.5）
    DRIBBLE_BACKOFF: float = 0.8  # 重置时退到球后方的距离（米，> 持球半径 0.2）

    # ----------------------------- 阵型站位 -----------------------------
    # 己方球门在 -x、进攻朝 +x，开球时全员位于己方半场（x<0）。
    # 比赛用 SevenVSevenField（55×36），后卫站位在己方禁区前沿之外（x>-18.5），
    # 保证默认只有守门员在禁区内，满足「非法防守」（禁区≤2 人）。
    BEAM_POSES: Mapping[type[Field], Mapping[int, tuple[float, float, float]]] = {
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
        # 旧 3v3 演示场地（9×14），保留兼容，非比赛用。
        HLAdultField: {
            1: (-6.3, 0.0, 0),
            2: (-4.5, 1.6, 0),
            3: (-4.5, -1.6, 0),
        },
        # 比赛场地（fifa7vs7，55×36，7v7）。号码 1 必须是守门员（规则要求）。
        SevenVSevenField: {
            1: (-25.5, 0.0, 0),   # 守门员（门前 2m）
            2: (-17.0, 7.0, 0),   # 左后卫（禁区外）
            3: (-17.0, -7.0, 0),  # 右后卫（禁区外）
            4: (-11.0, 10.0, 0),  # 左前卫
            5: (-13.0, 0.0, 0),   # 中前卫
            6: (-11.0, -10.0, 0), # 右前卫
            7: (-3.0, 0.0, 0),    # 前锋
        },
    }

    def __init__(self, agent):
        """
        创建与给定 agent 关联的 DecisionMaker。

        Args:
            agent: 拥有本 DecisionMaker 的主 agent。
        """
        from mujococodebase.agent import Agent  # type hinting

        self.agent: Agent = agent
        self.is_getting_up: bool = False
        # 带球违规持球规避所需的最小状态
        self._dribble_anchor: np.ndarray | None = None       # 本次持球起点
        self._dribble_reset_until: float | None = None       # 退开重置的截止服务器时间

    # ----------------------------- 主决策流程 -----------------------------

    def update_current_behavior(self) -> None:
        """
        决定 agent 在当前这一步应该做什么。
        """
        world = self.agent.world

        # 比赛结束，不再下发任何动作
        if world.playmode is PlayModeEnum.GAME_OVER:
            return

        playmode = world.playmode
        group = world.playmode_group

        # 开球前 / 进球后：把球员传送回各自的初始站位（含左右队镜像修复）
        if group in (PlayModeGroupEnum.ACTIVE_BEAM, PlayModeGroupEnum.PASSIVE_BEAM):
            self._beam_to(self.home_position(), rotation=0.0)

        # 倒地起身拥有最高优先级：只要正在起身或满足起身条件，先把人扶起来
        if self.is_getting_up or self.agent.skills_manager.is_ready(skill_name="GetUp"):
            self.is_getting_up = not self.agent.skills_manager.execute(skill_name="GetUp")
            self.agent.robot.commit_motor_targets_pd()
            return

        if playmode in (
            PlayModeEnum.BEFORE_KICK_OFF,
            PlayModeEnum.OUR_GOAL,
            PlayModeEnum.THEIR_GOAL,
        ):
            # 开球前与进球后：在站位上保持中性站立
            self.agent.skills_manager.execute("Neutral")

        elif self.is_goalkeeper():
            # 守门员：始终执行守门行为，不参与远端抢球
            self.goalkeeper_behavior()

        elif group is PlayModeGroupEnum.THEIR_KICK:
            # 对方死球（对方开球/任意球/角球/界外球/球门球等）：
            # 严禁抢先碰球，回防站位并与球保持安全距离
            self.defensive_positioning()

        elif playmode is PlayModeEnum.PLAY_ON or group is PlayModeGroupEnum.OUR_KICK:
            # 比赛进行中，或我方死球（由我方发球）：
            # 只有「离球最近的若干名外场球员」上前带球，其余人回到阵型站位
            if self.should_chase_ball():
                self.dribble_to_goal()
            else:
                self.move_to_formation()

        else:
            # 兜底：保持阵型站位
            self.move_to_formation()

        self.agent.robot.commit_motor_targets_pd()

    # ----------------------------- 角色与分工辅助 -----------------------------

    def is_goalkeeper(self) -> bool:
        """1 号球员为守门员（规则要求守门员球衣号必须为 1 号）。"""
        return self.agent.world.number == 1

    def _formation(self) -> Mapping[int, tuple[float, float, float]]:
        """当前场地对应的阵型站位表。"""
        return self.BEAM_POSES[type(self.agent.world.field)]

    def home_position(self, number: int | None = None) -> np.ndarray:
        """返回某号球员在当前场地下的初始站位（阵型基准点，2D）。"""
        if number is None:
            number = self.agent.world.number
        return np.array(self._formation()[number][:2], dtype=float)

    def _outfielders(self) -> list[int]:
        """全部外场球员号码（排除守门员 1 号）。"""
        return [n for n in self._formation() if n != 1]

    def _chasers_ordered(self) -> list[int]:
        """外场球员按「各自阵型站位到球的距离」升序排列（全 agent 结果一致）。"""
        ball = self.agent.world.ball_pos[:2]
        return sorted(
            self._outfielders(),
            key=lambda n: np.linalg.norm(self.home_position(n) - ball),
        )

    def should_chase_ball(self) -> bool:
        """判断「是否轮到我上前带球」：我在最近的 NUM_CHASERS 名外场球员之内。"""
        return self.agent.world.number in self._chasers_ordered()[: self.NUM_CHASERS]

    def _primary_chaser(self) -> int:
        """离球最近的那名外场球员（第一追球人）。"""
        return self._chasers_ordered()[0]

    def _box_defender(self) -> int:
        """
        被指定为「唯一可进入己方禁区的外场球员」（阵型站位最靠己方球门者）。

        守门员恒在禁区内算 1 人，再放行至多 1 名外场球员，即可由构造保证禁区
        内不超过 2 人，满足「非法防守」规则，且无需通信。
        """
        return min(self._outfielders(), key=lambda n: (self.home_position(n)[0], n))

    # ----------------------------- 站位与边界辅助 -----------------------------

    def _clamp_to_field(self, target: np.ndarray) -> np.ndarray:
        """把目标点夹在场地范围内（留 0.4m 边距）。"""
        field = self.agent.world.field
        half_x = field.get_length() / 2 - 0.4
        half_y = field.get_width() / 2 - 0.4
        return np.array([
            np.clip(target[0], -half_x, half_x),
            np.clip(target[1], -half_y, half_y),
        ])

    def _clamp_out_of_own_box(self, target: np.ndarray) -> np.ndarray:
        """若目标点落在己方禁区内，则把它推到禁区前沿之外（避免非法防守）。"""
        field = self.agent.world.field
        target = np.asarray(target, dtype=float)
        if field.is_in_own_penalty_area(target):
            goal_x = field.get_our_goal_position()[0]
            return np.array([goal_x + field.get_penalty_depth() + 0.1, target[1]])
        return target

    def _clamp_in_own_box(self, target: np.ndarray) -> np.ndarray:
        """把目标点夹在己方禁区内（守门员不得离开禁区）。"""
        field = self.agent.world.field
        goal_x = field.get_our_goal_position()[0]
        depth = field.get_penalty_depth()
        half_w = field.get_penalty_half_width()
        return np.array([
            np.clip(target[0], goal_x + 0.1, goal_x + depth - 0.1),
            np.clip(target[1], -half_w + 0.1, half_w - 0.1),
        ])

    def _guard_box(self, target: np.ndarray) -> np.ndarray:
        """带球时的禁区约束：只有指定盯防者或第一追球人可进入己方禁区。"""
        number = self.agent.world.number
        allowed = number == self._box_defender() or number == self._primary_chaser()
        return target if allowed else self._clamp_out_of_own_box(target)

    def _walk_or_stand(
        self, target: np.ndarray, face: np.ndarray | None, arrive_tol: float
    ) -> None:
        """走向目标点；到位后中性站立。face 给定时在临近目标处锁定朝向。"""
        my_pos = self.agent.world.global_position[:2]
        dist = np.linalg.norm(my_pos - target)
        if dist < arrive_tol:
            self.agent.skills_manager.execute("Neutral")
            return
        orientation = None
        if face is not None and dist <= self.ORIENT_NEAR:
            orientation = MathOps.vector_angle(face - my_pos)
        self.agent.skills_manager.execute(
            "Walk",
            target_2d=target,
            is_target_absolute=True,
            orientation=orientation,
        )

    # ----------------------------- 传送（含左右队镜像修复）-----------------------------

    def _beam_to(self, pos2d: np.ndarray, rotation: float) -> None:
        """
        传送到指定站位。

        修复重叠 bug：服务器按全局坐标解释 beam，而 agent 视角下右队坐标 = 全局
        取负。若直接下发会让左右两队传送到同一批全局坐标、同号球员重叠。这里对
        右队的坐标与朝向取反，保证落到己方半场。
        """
        world = self.agent.world
        x, y = float(pos2d[0]), float(pos2d[1])
        rot = rotation
        if not world.is_left_team:
            x, y = -x, -y
            rot = MathOps.normalize_deg(rotation + 180)
        self.agent.server.commit_beam(pos2d=(x, y), rotation=rot)

    # ----------------------------- 具体行为 -----------------------------

    def move_to_formation(self) -> None:
        """回到阵型站位（随球做攻防纵向平移）；到位后保持站立并面向球。"""
        world = self.agent.world
        ball = world.ball_pos[:2]
        home = self.home_position()

        # 攻防平移：球越靠对方半场越压上，越靠己方半场越退防
        shift = np.clip(
            ball[0] * self.FORMATION_SHIFT_K,
            -self.FORMATION_SHIFT_MAX,
            self.FORMATION_SHIFT_MAX,
        )
        target = self._clamp_to_field(np.array([home[0] + shift, home[1]]))

        # 非法防守：非指定盯防者不得进入己方禁区
        if world.number != self._box_defender():
            target = self._clamp_out_of_own_box(target)

        self._walk_or_stand(target, face=ball, arrive_tol=self.ARRIVE_TOL)

    def defensive_positioning(self) -> None:
        """
        对方死球时的防守站位：回到阵型站位，但绝不主动碰球。

        若站位点或自身当前位置距球过近（< 安全距离），则沿「远离球」方向后撤到
        安全距离外，避免抢先触球违规。
        """
        world = self.agent.world
        my_pos = world.global_position[:2]
        ball = world.ball_pos[:2]
        target = self.home_position()

        # 站位点离球太近：把目标点沿「球→站位点」方向推到安全距离外
        if np.linalg.norm(target - ball) < self.SAFE_DISTANCE:
            away = target - ball
            norm = np.linalg.norm(away)
            if norm < 1e-6:
                # 退化情况：朝己方球门方向后撤
                our_goal = np.array(world.field.get_our_goal_position(), dtype=float)
                away = our_goal - ball
                norm = np.linalg.norm(away) + 1e-9
            target = ball + away / norm * self.SAFE_DISTANCE

        # 我当前就站得离球太近：直接朝远离球方向后撤
        if np.linalg.norm(my_pos - ball) < self.SAFE_DISTANCE:
            away = my_pos - ball
            norm = np.linalg.norm(away)
            away = away / norm if norm > 1e-6 else np.array([-1.0, 0.0])
            target = ball + away * self.SAFE_DISTANCE

        target = self._clamp_to_field(target)
        if world.number != self._box_defender():
            target = self._clamp_out_of_own_box(target)

        self._walk_or_stand(target, face=ball, arrive_tol=self.ARRIVE_TOL)

    def goalkeeper_behavior(self) -> None:
        """
        守门员行为。

        默认在球门线前横向跟球封角（x 固定门前、y 跟随球并限制在门柱范围内，
        始终面向球）。仅当比赛进行（PLAY_ON）且球进入己方禁区时，才上前把球往
        前顶离危险区。**全程不离开禁区**（避免站位违规、并满足点球规则）。
        """
        world = self.agent.world
        field = world.field
        ball = world.ball_pos[:2]
        goal_x = field.get_our_goal_position()[0]

        # 解围：仅在比赛进行且球进入己方禁区时，上前把球往前顶（朝 +x）
        if world.playmode is PlayModeEnum.PLAY_ON and field.is_in_own_penalty_area(ball):
            target = self._clamp_in_own_box(ball)
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=target,
                is_target_absolute=True,
                orientation=0.0,  # 面向对方球门方向，把球顶离危险区
            )
            return

        # 横向拦截站位（夹在禁区内）
        target = self._clamp_in_own_box(
            np.array([
                goal_x + self.GK_LINE_OFFSET,
                np.clip(ball[1], -self.GK_HALF_WIDTH, self.GK_HALF_WIDTH),
            ])
        )
        self._walk_or_stand(target, face=ball, arrive_tol=self.GK_ARRIVE_TOL)

    def dribble_to_goal(self) -> None:
        """
        带球朝对方球门推进，并规避「违规持球」。

        规则：球在 0.2m 内、且持球期间移动 ≥1.0m 即判违规持球（不论对手位置）。
        因此本方法：① 站到球后方一点，把球往对方球门方向小幅顶推，自然变成连续
        点球而非长抱；② 用锚点记录本次持球起点，一旦推进超过 DRIBBLE_MAX，就退
        开 DRIBBLE_RESET_TIME（> 规则的 0.5s 重置时间）让持球计时清零再继续。
        """
        world = self.agent.world
        ball = world.ball_pos[:2].astype(float)
        my_pos = world.global_position[:2]
        their_goal = np.array(world.field.get_their_goal_position()[:2], dtype=float)
        now = world.server_time

        ball_to_goal = their_goal - ball
        bg_norm = np.linalg.norm(ball_to_goal)
        if bg_norm < 1e-6:
            self.agent.skills_manager.execute("Neutral")
            return
        bg_dir = ball_to_goal / bg_norm

        # 处于退开冷却中：退到球后方远处（不触球），让持球计时重置
        if (
            self._dribble_reset_until is not None
            and now is not None
            and now < self._dribble_reset_until
        ):
            backoff = self._guard_box(ball - bg_dir * self.DRIBBLE_BACKOFF)
            self._walk_or_stand(backoff, face=ball, arrive_tol=self.ARRIVE_TOL)
            return

        # 冷却结束：以当前球位为新锚点，恢复带球
        if (
            self._dribble_reset_until is not None
            and now is not None
            and now >= self._dribble_reset_until
        ):
            self._dribble_anchor = ball.copy()
            self._dribble_reset_until = None

        if self._dribble_anchor is None:
            self._dribble_anchor = ball.copy()

        # 推进过头：进入退开冷却
        advanced = np.linalg.norm(ball - self._dribble_anchor)
        if advanced >= self.DRIBBLE_MAX and now is not None:
            self._dribble_reset_until = now + self.DRIBBLE_RESET_TIME
            backoff = self._guard_box(ball - bg_dir * self.DRIBBLE_BACKOFF)
            self._walk_or_stand(backoff, face=ball, arrive_tol=self.ARRIVE_TOL)
            return

        # 正常推进：走到球后方一点，朝对方球门把球往前顶
        carry = self._guard_box(ball - bg_dir * self.DRIBBLE_BEHIND)
        far = np.linalg.norm(my_pos - carry) > self.ORIENT_NEAR
        self.agent.skills_manager.execute(
            "Walk",
            target_2d=carry,
            is_target_absolute=True,
            orientation=None if far else MathOps.vector_angle(ball_to_goal),
        )
