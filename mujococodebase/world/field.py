from abc import ABC, abstractmethod
from typing import override
from mujococodebase.world.field_landmarks import FieldLandmarks


class Field(ABC):
    def __init__(self, world):
        from mujococodebase.world.world import World  # type hinting
        self.world: World = world
        self.field_landmarks: FieldLandmarks = FieldLandmarks(world=self.world)
    
    def get_our_goal_position(self):
        return (-self.get_length()/2, 0)

    def get_their_goal_position(self):
        return (self.get_length()/2, 0)

    # ----------------------------- 禁区近似 -----------------------------
    # 比赛规则约束「本方禁区最多 2 人」（非法防守），需要判断某点是否在己方禁区。
    # 服务器未通过感知暴露禁区尺寸，这里按 9×14 场地给出近似值，留成方法便于按
    # 真实服务器实测后调整。

    def get_penalty_depth(self) -> float:
        """禁区从己方端线向场内延伸的纵深（米）。"""
        return 1.8

    def get_penalty_half_width(self) -> float:
        """禁区沿球门宽度方向的半幅（米）。"""
        return 2.2

    def is_in_own_penalty_area(self, pos2d) -> bool:
        """判断 2D 点是否落在己方禁区内（agent 视角：己方球门在 -x）。"""
        goal_x = self.get_our_goal_position()[0]  # 位于 -L/2
        # 己方禁区在 x ∈ [goal_x, goal_x + depth]，|y| ≤ half_width
        in_x = goal_x <= pos2d[0] <= goal_x + self.get_penalty_depth()
        in_y = abs(pos2d[1]) <= self.get_penalty_half_width()
        return in_x and in_y

    @abstractmethod
    def get_width(self):
        raise NotImplementedError()

    @abstractmethod
    def get_length(self):
        raise NotImplementedError()
    

class FIFAField(Field):
    def __init__(self, world):
        super().__init__(world)
    
    @override
    def get_width(self):
        return 68
    
    @override
    def get_length(self):
        return 105
    

class HLAdultField(Field):
    def __init__(self, world):
        super().__init__(world)

    @override
    def get_width(self):
        return 9

    @override
    def get_length(self):
        return 14


class SevenVSevenField(Field):
    def __init__(self, world):
        super().__init__(world)

    @override
    def get_width(self):
        return 36

    @override
    def get_length(self):
        return 55