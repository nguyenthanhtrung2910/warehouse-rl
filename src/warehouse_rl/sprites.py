from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pygame
from pygame.math import Vector2

from warehouse_rl.warehouse import NODE_SIZE, LineNode, RayNode, Warehouse

DEFAULT_REWARD = 0
FRAME_PER_STEP = 5


class Action(Enum):
    # NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


@dataclass
class StepResult:
    is_performed: bool
    reward: float


class Shuttle(pygame.sprite.Sprite):
    pos: RayNode
    parcel: Parcel | None
    image: pygame.Surface
    rect: pygame.Rect
    parcel_sprites: pygame.sprite.Group

    def __init__(self, pos: RayNode, parcel_sprites: pygame.sprite.Group):
        super().__init__()
        self.pos = pos
        self.pos.robot = self
        self.parcel = None
        self.__set_image()
        self.rect = self.image.get_rect()
        self.rect.center = self.pos.world_pos
        self.parcel_sprites = parcel_sprites

    def __set_image(self):
        self.image = pygame.Surface(NODE_SIZE, pygame.SRCALPHA)
        pygame.draw.circle(
            self.image,
            (255, 0, 0),
            NODE_SIZE / 2,
            min(NODE_SIZE) / 2,
        )

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
        rect.center = self.pos.world_pos
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
        return DEFAULT_REWARD

    def move_down(self):
        self.pos.robot = None
        self.pos = self.pos.down
        self.pos.robot = self
        return DEFAULT_REWARD

    def move_left(self):
        self.pos.robot = None
        self.pos = self.pos.left
        self.pos.robot = self
        return DEFAULT_REWARD

    def move_right(self):
        self.pos.robot = None
        self.pos = self.pos.right
        self.pos.robot = self
        return DEFAULT_REWARD

    def pick_up(self):
        if self.pos.from_line and self.pos.from_line.isPalletize and not self.parcel:
            self.parcel = self.pos.from_line.parcel
            self.pos.from_line.parcel = Parcel(self.pos.from_line)
            self.parcel_sprites.add(self.pos.from_line.parcel)
            return self.pos.from_line
        return None

    def drop_off(self):
        if self.pos.to_line and not self.pos.to_line.parcel and self.parcel:
            current = self.pos.to_line
            # loop until find a next line node that has
            while current.next_node:
                if current.next_node.parcel:
                    break
                current = current.next_node
            current.parcel = self.parcel
            self.parcel = None
            return current
        return None

    def step(self, action: int, renderer: Warehouse | None = None):
        # check if action is legal
        # actually, action from agent always is legal because of action mask
        # we check for case that all actions are illegal
        is_action_legal = self.__is_legal_move(action)
        if not is_action_legal:
            # if no action is legal, do nothing
            return StepResult(False, DEFAULT_REWARD)
        reward = 0
        if action == Action.UP:
            reward = self.move_up()
        elif action == Action.DOWN:
            reward = self.move_down()
        elif action == Action.LEFT:
            reward = self.move_left()
        elif action == Action.RIGHT:
            reward = self.move_right()
        else:
            raise ValueError("Invalid input action.")
        if renderer:
            diff = Vector2(self.next_rect.center) - self.rect.center
            for _ in range(0, FRAME_PER_STEP):
                self.rect.move_ip(diff / FRAME_PER_STEP)
                if self.parcel:
                    self.parcel.rect.move_ip(diff / FRAME_PER_STEP)
                renderer.render()
            line_node = self.pick_up()
            if line_node:
                diff = self.rect.center - line_node.world_pos
                for _ in range(0, FRAME_PER_STEP):
                    self.parcel.rect.move_ip(diff / FRAME_PER_STEP)
                    renderer.render()
            line_node = self.drop_off()
            if line_node:
                diff = line_node.world_pos - self.rect.center
                for _ in range(0, FRAME_PER_STEP):
                    line_node.parcel.rect.move_ip(diff / FRAME_PER_STEP)
                    renderer.render()
        return StepResult(True, reward)


class Parcel(pygame.sprite.Sprite):
    image: pygame.Surface
    rect: pygame.Rect

    def __init__(self, pos: LineNode) -> None:
        super().__init__()
        pos.parcel = self
        self.image = pygame.Surface(NODE_SIZE, pygame.SRCALPHA)
        pygame.draw.circle(
            self.image,
            (0, 255, 0),
            NODE_SIZE / 2,
            min(NODE_SIZE) / 4,
        )
        self.rect = self.image.get_rect()
        self.rect.center = pos.world_pos
