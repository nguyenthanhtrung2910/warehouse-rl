from __future__ import annotations

import numpy as np
import pygame
from pygame.math import Vector2

from warehouse_rl import warehouse
from warehouse_rl.enums import NODE_SIZE, Action

FRAME_PER_STEP = 5


class Parcel:
    image: pygame.Surface
    rect: pygame.Rect

    def __init__(self, pos: warehouse.LineNode):
        pos.parcel = self
        self.image = pygame.Surface(NODE_SIZE, pygame.SRCALPHA)
        pygame.draw.circle(
            self.image,
            (0, 255, 0),
            NODE_SIZE / 2,
            min(NODE_SIZE) / 4,
        )
        self.rect = self.image.get_rect()
        self.rect.center = pos.world_pos  # type: ignore

    @property
    def world_pos(self):
        return Vector2(self.rect.center)

    @world_pos.setter
    def world_pos(self, world_pos: Vector2):
        self.rect.center = world_pos  # type: ignore

    def world_translate(self, bias: Vector2):
        self.rect.move_ip(bias)

    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)


class Shuttle:
    map_size: Vector2
    image: pygame.Surface
    pos: warehouse.RayNode
    parcel: Parcel | None
    rect: pygame.Rect

    def __init__(
        self,
        pos: warehouse.RayNode,
        map_size: Vector2,
    ):
        self.map_size = map_size
        self.pos = pos
        self.pos.robot = self
        self.parcel = None
        self.image = pygame.Surface(NODE_SIZE, pygame.SRCALPHA)
        pygame.draw.circle(
            self.image,
            (255, 0, 0),
            NODE_SIZE / 2,
            min(NODE_SIZE) / 2,
        )
        self.rect = self.image.get_rect()
        self.rect.center = pos.world_pos  # type: ignore

    @property
    def world_pos(self):
        return Vector2(self.rect.center)
    
    @world_pos.setter
    def world_pos(self, world_pos: Vector2):
        self.rect.center = world_pos  # type: ignore
        if self.parcel:
            self.parcel.world_pos = world_pos
    
    def world_translate(self, bias: Vector2):
        self.rect.move_ip(bias)
        if self.parcel:
            self.parcel.world_translate(bias)

    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)
        if self.parcel:
            self.parcel.draw(screen)

    @property
    def mask(self):
        return np.array(
            [self.__is_legal_move(action) for action in Action], dtype=np.uint8
        )

    @property
    def state(self):
        has_parcel = 1 if self.parcel else 0
        return np.array(
            [
                self.pos.x / self.map_size.x,
                self.pos.y / self.map_size.y,
                has_parcel,
            ],
            dtype=np.float32,
        )

    def pick_up(self):
        if (
            self.pos.from_line
            and self.pos.from_line.isPalletize
            and self.pos.from_line.parcel
            and not self.parcel
        ):
            self.parcel = self.pos.from_line.parcel
            self.pos.from_line.parcel = Parcel(self.pos.from_line)
            return self.parcel
        return None

    def drop_off(self):
        if self.pos.to_line and not self.pos.to_line.parcel and self.parcel:
            current = self.pos.to_line
            # Loop until find a next line node that already has parcel
            while current.next_node:
                if current.next_node.parcel:
                    break
                current = current.next_node
            current.parcel = self.parcel
            self.parcel = None
            return current, current.parcel
        return None, None

    def reset(self, pos: warehouse.RayNode):
        self.pos.robot = None
        self.pos = pos
        self.pos.robot = self
        self.parcel = None
        self.rect = self.image.get_rect()
        self.rect.center = pos.world_pos  # type: ignore

    def step(self, action: Action):
        # Check if action is legal
        # Actually, action from agent always is legal because of action mask,
        # we check for case that all actions are illegal
        is_action_legal = self.__is_legal_move(action)
        if not is_action_legal:
            # If no action is legal, do nothing
            return False
        match action:
            case Action.Up:
                self.__move_up()
            case Action.Down:
                self.__move_down()
            case Action.Left:
                self.__move_left()
            case Action.Right:
                self.__move_right()
            case _:
                raise ValueError(f"Invalid action value {action}.")
        return True
    
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

    def __move_up(self):
        self.pos.robot = None
        self.pos = self.pos.up  # pyright: ignore[reportAttributeAccessIssue]
        self.pos.robot = self

    def __move_down(self):
        self.pos.robot = None
        self.pos = self.pos.down  # pyright: ignore[reportAttributeAccessIssue]
        self.pos.robot = self

    def __move_left(self):
        self.pos.robot = None
        self.pos = self.pos.left  # pyright: ignore[reportAttributeAccessIssue]
        self.pos.robot = self

    def __move_right(self):
        self.pos.robot = None
        self.pos = self.pos.right  # pyright: ignore[reportAttributeAccessIssue]
        self.pos.robot = self
