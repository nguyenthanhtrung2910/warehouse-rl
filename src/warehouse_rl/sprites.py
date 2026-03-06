from __future__ import annotations
from enum import Enum
import numpy as np
from warehouse_rl.warehouse import RayNode
import pygame


class Action(Enum):
    # NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Shuttle(pygame.sprite.Sprite):
    pos: RayNode

    def __init__(self, pos: RayNode):
        self.pos = pos

    def __is_legal_move(self, act: Action):
        if act == Action.UP:
            if not self.pos.up:
                return False
            if self.pos.up.robot:
                return False
        if act == Action.DOWN:
            if not self.pos.down:
                return False
            if self.pos.down.robot:
                return False
        if act == Action.LEFT:
            if not self.pos.left:
                return False
            if self.pos.left.robot:
                return False
        if act == Action.RIGHT:
            if not self.pos.right:
                return False
            if self.pos.right.robot:
                return False
        return True

    @property
    def mask(self):
        return np.array(
            [self.__is_legal_move(action) for action in Action], dtype=np.uint8
        )
