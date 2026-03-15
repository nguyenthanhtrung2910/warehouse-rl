from __future__ import annotations

import numpy as np
import pygame
from pygame.math import Vector2

from warehouse_rl import warehouse
from warehouse_rl.enums import NODE_SIZE, Action

FRAME_PER_STEP = 5


class Parcel:
    __image: pygame.Surface
    __rect: pygame.Rect

    def __init__(self, pos: warehouse.LineNode):
        pos.parcel = self
        self.__image = pygame.Surface(NODE_SIZE, pygame.SRCALPHA)
        pygame.draw.circle(
            self.__image,
            (0, 255, 0),
            NODE_SIZE / 2,
            min(NODE_SIZE) / 4,
        )
        self.__rect = self.__image.get_rect()
        self.__rect.center = pos.world_pos  # type: ignore

    def translate(self, bias: Vector2):
        self.__rect.move_ip(bias)

    def draw(self, screen: pygame.Surface):
        screen.blit(self.__image, self.__rect)

    @property
    def rect(self):
        return self.__rect

    def set_rect_center(self, coorddinate: Vector2):
        self.__rect.center = coorddinate  # type: ignore


class Shuttle:
    __map_size: Vector2
    __pos: warehouse.RayNode
    __parcel: Parcel | None
    __image: pygame.Surface
    __rect: pygame.Rect

    def __init__(
        self,
        pos: warehouse.RayNode,
        map_size: Vector2,
    ):
        self.__map_size = map_size
        self.__pos = pos
        self.__pos.robot = self
        self.__parcel = None
        self.__image = pygame.Surface(NODE_SIZE, pygame.SRCALPHA)
        pygame.draw.circle(
            self.__image,
            (255, 0, 0),
            NODE_SIZE / 2,
            min(NODE_SIZE) / 2,
        )
        self.__rect = self.__image.get_rect()
        self.__rect.center = pos.world_pos  # type: ignore

    def translate(self, bias: Vector2):
        self.__rect.move_ip(bias)
        if self.__parcel:
            self.__parcel.translate(bias)

    def draw(self, screen: pygame.Surface):
        screen.blit(self.__image, self.__rect)
        if self.__parcel:
            self.__parcel.draw(screen)

    @property
    def rect(self):
        return self.__rect

    def set_rect_center(self, coorddinate: Vector2):
        self.__rect.center = coorddinate  # type: ignore
        if self.__parcel:
            self.__parcel.set_rect_center(coorddinate)

    @property
    def next_rect_center(self):
        return self.__pos.world_pos

    @property
    def mask(self):
        return np.array(
            [self.__is_legal_move(action) for action in Action], dtype=np.uint8
        )

    @property
    def state(self):
        has_parcel = 1 if self.__parcel else 0
        return np.array(
            [
                self.__pos.x / self.__map_size.x,
                self.__pos.y / self.__map_size.y,
                has_parcel,
            ],
            dtype=np.float32,
        )

    def __is_legal_move(self, act: Action):
        match act:
            case Action.Up:
                if not self.__pos.up:
                    return False
                if self.__pos.up.robot:
                    return False
            case Action.Down:
                if not self.__pos.down:
                    return False
                if self.__pos.down.robot:
                    return False
            case Action.Left:
                if not self.__pos.left:
                    return False
                if self.__pos.left.robot:
                    return False
            case Action.Right:
                if not self.__pos.right:
                    return False
                if self.__pos.right.robot:
                    return False
            case _:
                raise ValueError(f"Invalid action value {act}.")
        return True

    def __move_up(self):
        self.__pos.robot = None
        self.__pos = self.__pos.up  # pyright: ignore[reportAttributeAccessIssue]
        self.__pos.robot = self

    def __move_down(self):
        self.__pos.robot = None
        self.__pos = self.__pos.down  # pyright: ignore[reportAttributeAccessIssue]
        self.__pos.robot = self

    def __move_left(self):
        self.__pos.robot = None
        self.__pos = self.__pos.left  # pyright: ignore[reportAttributeAccessIssue]
        self.__pos.robot = self

    def __move_right(self):
        self.__pos.robot = None
        self.__pos = self.__pos.right  # pyright: ignore[reportAttributeAccessIssue]
        self.__pos.robot = self

    def pick_up(self):
        if (
            self.__pos.from_line
            and self.__pos.from_line.isPalletize
            and self.__pos.from_line.parcel
            and not self.__parcel
        ):
            self.__parcel = self.__pos.from_line.parcel
            self.__pos.from_line.parcel = Parcel(self.__pos.from_line)
            return self.__parcel
        return None

    def drop_off(self):
        if self.__pos.to_line and not self.__pos.to_line.parcel and self.__parcel:
            current = self.__pos.to_line
            # Loop until find a next line node that already has parcel
            while current.next_node:
                if current.next_node.parcel:
                    break
                current = current.next_node
            current.parcel = self.__parcel
            self.__parcel = None
            return current, current.parcel
        return None, None

    def reset(self, pos: warehouse.RayNode):
        self.__pos.robot = None
        self.__pos = pos
        self.__pos.robot = self
        self.__parcel = None
        self.__rect = self.__image.get_rect()
        self.__rect.center = pos.world_pos  # type: ignore

    def step(self, action: Action):
        # Check if action is legal
        # Actually, action from agent always is legal because of action mask,
        # we check for case that all actions are illegal
        is_action_legal = self.__is_legal_move(action)
        if not is_action_legal:
            # if no action is legal, do nothing
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
