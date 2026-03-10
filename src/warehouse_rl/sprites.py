from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame
from pygame.math import Vector2

from warehouse_rl.enums import Action, RenderMode
from warehouse_rl.warehouse import NODE_SIZE, LineNode, RayNode, Warehouse

DEFAULT_REWARD = -0.1
PICKUP_REWARD = 1
DROPOFF_REWARD = 5
FRAME_PER_STEP = 5


@dataclass
class StepResult:
    is_performed: bool
    reward: float


class Shuttle(pygame.sprite.Sprite):
    pos: RayNode
    parcel: Parcel | None
    render_mode: RenderMode
    image: pygame.Surface | None
    rect: pygame.Rect | None

    def __init__(
        self,
        pos: RayNode,
        render_mode: RenderMode = RenderMode.NoRender,
    ):
        super().__init__()
        self.pos = pos
        self.pos.robot = self
        self.parcel = None
        self.render_mode = render_mode
        match render_mode:
            case RenderMode.Human:
                self.image = pygame.Surface(NODE_SIZE, pygame.SRCALPHA)
                pygame.draw.circle(
                    self.image,
                    (255, 0, 0),
                    NODE_SIZE / 2,
                    min(NODE_SIZE) / 2,
                )
                self.rect = self.image.get_rect()
                self.rect.center = self.pos.world_pos
            case RenderMode.NoRender:
                self.image = None
                self.rect = None
            case _:
                raise ValueError(f"Invalid render_mode value {render_mode}")

    def __is_legal_move(self, act: Action):
        match act:
            case Action.Up:
                if not self.pos.up:
                    return False
                if self.pos.up.robot:
                    return False
            case Action.Down:
                if not self.pos.down:
                    return False
                if self.pos.down.robot:
                    return False
            case Action.Left:
                if not self.pos.left:
                    return False
                if self.pos.left.robot:
                    return False
            case Action.Right:
                if not self.pos.right:
                    return False
                if self.pos.right.robot:
                    return False
            case _:
                raise ValueError(f"Invalid action value {act}.")
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

    def move_down(self):
        self.pos.robot = None
        self.pos = self.pos.down
        self.pos.robot = self

    def move_left(self):
        self.pos.robot = None
        self.pos = self.pos.left
        self.pos.robot = self

    def move_right(self):
        self.pos.robot = None
        self.pos = self.pos.right
        self.pos.robot = self

    def pick_up(self):
        if self.pos.from_line and self.pos.from_line.isPalletize and not self.parcel:
            self.parcel = self.pos.from_line.parcel
            self.pos.from_line.parcel = Parcel(self.pos.from_line, self.render_mode)
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

    def step(self, action: int, renderer: Warehouse):
        # check if action is legal
        # actually, action from agent always is legal because of action mask
        # we check for case that all actions are illegal
        reward = DEFAULT_REWARD
        is_action_legal = self.__is_legal_move(action)
        if not is_action_legal:
            # if no action is legal, do nothing
            return StepResult(False, reward)
        match action:
            case Action.Up:
                self.move_up()
            case Action.Down:
                self.move_down()
            case Action.Left:
                self.move_left()
            case Action.Right:
                self.move_right()
            case _:
                raise ValueError(f"Invalid action value {action}.")
        # simulate movement
        if self.render_mode == RenderMode.Human:
            diff = Vector2(self.next_rect.center) - self.rect.center
            for _ in range(0, FRAME_PER_STEP):
                self.rect.move_ip(diff / FRAME_PER_STEP)
                if self.parcel:
                    self.parcel.rect.move_ip(diff / FRAME_PER_STEP)
                renderer.render()
        # try to pick up
        line_node = self.pick_up()
        if line_node:
            reward = PICKUP_REWARD
            # simulate pick up
            if self.render_mode == RenderMode.Human:
                diff = self.rect.center - line_node.world_pos
                for _ in range(0, FRAME_PER_STEP):
                    self.parcel.rect.move_ip(diff / FRAME_PER_STEP)
                    renderer.render()
                renderer.parcel_sprites.add(self.pos.from_line.parcel)
        # try to drop off
        line_node = self.drop_off()
        if line_node:
            reward = DROPOFF_REWARD
            # simulate drop off
            if self.render_mode == RenderMode.Human:
                diff = line_node.world_pos - self.rect.center
                for _ in range(0, FRAME_PER_STEP):
                    line_node.parcel.rect.move_ip(diff / FRAME_PER_STEP)
                    renderer.render()
        return StepResult(True, reward)


class Parcel(pygame.sprite.Sprite):
    image: pygame.Surface | None
    rect: pygame.Rect | None

    def __init__(
        self, pos: LineNode, render_mode: RenderMode = RenderMode.NoRender
    ) -> None:
        super().__init__()
        pos.parcel = self
        match render_mode:
            case RenderMode.Human:
                self.image = pygame.Surface(NODE_SIZE, pygame.SRCALPHA)
                pygame.draw.circle(
                    self.image,
                    (0, 255, 0),
                    NODE_SIZE / 2,
                    min(NODE_SIZE) / 4,
                )
                self.rect = self.image.get_rect()
                self.rect.center = pos.world_pos
            case RenderMode.NoRender:
                self.image = None
                self.rect = None
            case _:
                raise ValueError(f"Invalid render_mode value {render_mode}")
