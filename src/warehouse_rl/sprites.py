from __future__ import annotations

from enum import Enum

import numpy as np
import pygame

from warehouse_rl.warehouse import NODE_SIZE, RayNode

DEFAULT_REWARD = 0


class Action(Enum):
    # NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Shuttle(pygame.sprite.Sprite):
    pos: RayNode
    image: pygame.Surface
    rect: pygame.Rect

    def __init__(self, pos: RayNode):
        super().__init__()
        self.pos = pos
        self.pos.robot = self
        self.__set_image()
        self.rect = self.image.get_rect()
        self.rect.center = (
            (self.pos.x + 0.5) * NODE_SIZE[0],
            (self.pos.y + 0.5) * NODE_SIZE[1],
        )

    def __set_image(self):
        self.image = pygame.Surface(NODE_SIZE, pygame.SRCALPHA)
        pygame.draw.circle(
            self.image,
            color=(255, 0, 0),
            center=(NODE_SIZE[0] / 2, NODE_SIZE[1] / 2),
            radius=min(NODE_SIZE) / 2,
        )
        # pygame.draw.rect(self.image, (0, 0, 0), self.image.get_rect(), 1)

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
    def next_rect(self) -> pygame.Rect:
        rect = self.image.get_rect()
        rect.center = (
            (self.pos.x + 0.5) * NODE_SIZE[0],
            (self.pos.y + 0.5) * NODE_SIZE[1],
        )
        return rect

    @property
    def mask(self):
        return np.array(
            [self.__is_legal_move(action) for action in Action], dtype=np.uint8
        )

    def move_up(self):
        self.pos.robot = None
        self.pos = self.pos.up
        self.pos.robot = self
        return True

    def move_down(self):
        self.pos.robot = None
        self.pos = self.pos.down
        self.pos.robot = self
        return True

    def move_left(self):
        self.pos.robot = None
        self.pos = self.pos.left
        self.pos.robot = self
        return True

    def move_right(self):
        self.pos.robot = None
        self.pos = self.pos.right
        self.pos.robot = self
        return True

    def step(self, action: int) -> tuple[bool, float]:
        # check if action is legal
        # actually, action from agent always is legal because of action mask
        # we check for case that all actions are illegal
        is_action_legal = self.__is_legal_move(action)
        if not is_action_legal:
            # if no action is legal, do nothing
            return False, DEFAULT_REWARD
        if action == Action.UP:
            return self.move_up()
        if action == Action.DOWN:
            return self.move_down()
        if action == Action.LEFT:
            return self.move_left()
        if action == Action.RIGHT:
            return self.move_right()
