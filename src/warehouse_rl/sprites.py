from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import numpy as np
import numpy.typing as npt
import pygame
from pygame.math import Vector2

from warehouse_rl import map, warehouse
from warehouse_rl.enums import NODE_SIZE, Action


@dataclass
class StepResult:
    reward: float
    movements: list[warehouse.Movement] | None


class Sprite(ABC):
    image: pygame.Surface
    rect: pygame.Rect

    def __init__(self):
        self.image = pygame.Surface(NODE_SIZE, pygame.SRCALPHA)
        self.rect = self.image.get_rect()

    @property
    def world_pos(self):
        return Vector2(self.rect.center)

    @world_pos.setter
    @abstractmethod
    def world_pos(self, world_pos: Vector2):
        pass

    @abstractmethod
    def world_translate(self, bias: Vector2):
        pass

    @abstractmethod
    def draw(self, screen: pygame.Surface):
        pass


class Parcel(Sprite):
    __is_requested: bool

    def __init__(self, pos: map.LineNode, is_requested: bool = False):
        super().__init__()
        pos.parcel = self
        color = (158, 52, 235) if is_requested else (0, 255, 0)
        pygame.draw.circle(
            self.image,
            color,
            NODE_SIZE / 2,
            min(NODE_SIZE) / 4,
        )
        self.rect.center = pos.world_pos  # pyright: ignore[reportAttributeAccessIssue]
        self.__is_requested = is_requested

    @Sprite.world_pos.setter
    def world_pos(self, world_pos: Vector2):
        self.rect.center = world_pos  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def world_translate(self, bias: Vector2):
        self.rect.move_ip(bias)

    @override
    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)

    @property
    def is_requested(self):
        return self.__is_requested

    @is_requested.setter
    def is_requested(self, is_requested: bool):
        self.__is_requested = is_requested
        color = (158, 52, 235) if is_requested else (0, 255, 0)
        pygame.draw.circle(
            self.image,
            color,
            NODE_SIZE / 2,
            min(NODE_SIZE) / 4,
        )


class Shuttle(Sprite):
    DEFAULT_REWARD = -0.1
    map_size: Vector2
    pos: map.RayNode
    parcel: Parcel | None

    def __init__(
        self,
        pos: map.RayNode,
        map_size: Vector2,
    ):
        super().__init__()
        self.map_size = map_size
        self.pos = pos
        self.pos.robot = self
        self.parcel = None
        self.rect.center = pos.world_pos  # pyright: ignore[reportAttributeAccessIssue]

    @abstractmethod
    def pick_up(self) -> StepResult:
        pass

    @abstractmethod
    def drop_off(self) -> StepResult:
        pass

    @Sprite.world_pos.setter
    def world_pos(self, world_pos: Vector2):
        self.rect.center = world_pos  # pyright: ignore[reportAttributeAccessIssue]
        if self.parcel:
            self.parcel.world_pos = world_pos

    @override
    def world_translate(self, bias: Vector2):
        self.rect.move_ip(bias)
        if self.parcel:
            self.parcel.world_translate(bias)

    @override
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
    @abstractmethod
    def state(self) -> npt.NDArray[np.float32]:
        pass

    def reset(self, pos: map.RayNode):
        self.pos.robot = None
        self.pos = pos
        self.pos.robot = self
        self.parcel = None
        self.world_pos = pos.world_pos

    def step(self, action: Action):
        # Check if action is legal
        # Actually, action from agent always is legal because of action mask,
        # we check for case that all actions are illegal
        is_action_legal = self.__is_legal_move(action)
        if not is_action_legal:
            # If no action is legal, do nothing
            return StepResult(0.0, None)
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
        return StepResult(
            Shuttle.DEFAULT_REWARD, [warehouse.Movement(self, self.pos.world_pos)]
        )

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


class Loader(Shuttle):
    PICKUP_REWARD = 1.0
    DROPOFF_REWARD = 5.0

    def __init__(
        self,
        pos: map.RayNode,
        map_size: Vector2,
    ):
        super().__init__(pos, map_size)
        pygame.draw.circle(
            self.image,
            (255, 0, 0),
            NODE_SIZE / 2,
            min(NODE_SIZE) / 2,
        )

    @override
    def pick_up(self):
        if (
            self.pos.from_line
            and self.pos.from_line.is_depalletized
            and self.pos.from_line.parcel
            and not self.parcel
        ):
            self.parcel = self.pos.from_line.parcel
            self.pos.from_line.parcel = Parcel(self.pos.from_line)
            return StepResult(
                Loader.PICKUP_REWARD, [warehouse.Movement(self.parcel, self.world_pos)]
            )
        return StepResult(0.0, None)

    @override
    def drop_off(self):
        if (
            self.pos.to_line
            and not self.pos.to_line.parcel
            and self.parcel
            and not self.pos.to_line.is_palletized
        ):
            current = self.pos.to_line
            # Loop until find a next line node that already has parcel
            while current.next_node:
                if current.next_node.parcel:
                    break
                current = current.next_node
            current.parcel = self.parcel
            self.parcel = None
            return StepResult(
                Loader.DROPOFF_REWARD,
                [warehouse.Movement(current.parcel, current.world_pos)],
            )
        return StepResult(0.0, None)
    
    @property
    @override
    def state(self):
        has_parcel = 1.0 if self.parcel else 0.0
        return np.array(
            [
                self.pos.x / self.map_size.x,
                self.pos.y / self.map_size.y,
                has_parcel,
            ],
            dtype=np.float32,
        )


class Picker(Shuttle):
    PICK_UP_REWARD = 1.0
    PICK_UP_REQ_REWARD = 2.0
    DROP_OFF_REWARD = 5.0
    DROP_OFF_REQ_REWARD = 10.0

    def __init__(
        self,
        pos: map.RayNode,
        map_size: Vector2,
    ):
        super().__init__(pos, map_size)
        pygame.draw.circle(
            self.image,
            (235, 119, 52),
            NODE_SIZE / 2,
            min(NODE_SIZE) / 2,
        )

    @override
    def pick_up(self):
        if (
            self.pos.from_line
            and not self.pos.from_line.is_depalletized
            and self.pos.from_line.parcel
            and not self.parcel
            and self.__has_requested(self.pos.from_line)
        ):
            movements: list[warehouse.Movement] = []
            self.parcel = self.pos.from_line.parcel
            self.pos.from_line.parcel = None
            movements.append(warehouse.Movement(self.parcel, self.world_pos))
            current = self.pos.from_line
            # Loop until no previous node or previous node has no parcel
            while current.previous_node and current.previous_node.parcel:
                current.parcel = current.previous_node.parcel
                current.previous_node.parcel = None
                movements.append(warehouse.Movement(current.parcel, current.world_pos))
                current = current.previous_node
            if self.parcel.is_requested:
                return StepResult(Picker.PICK_UP_REQ_REWARD, movements)
            else:
                return StepResult(Picker.PICK_UP_REWARD, movements)
        return StepResult(0.0, None)

    @override
    def drop_off(self):
        if self.pos.to_line and not self.pos.to_line.parcel and self.parcel:
            if self.pos.to_line.is_palletized:
                if self.parcel.is_requested:
                    self.pos.to_line.parcel = self.parcel
                    self.parcel = None
                    return StepResult(
                        Picker.DROP_OFF_REQ_REWARD,
                        [
                            warehouse.Movement(
                                self.pos.to_line.parcel, self.pos.to_line.world_pos
                            )
                        ],
                    )
            else:
                if not self.parcel.is_requested:
                    current = self.pos.to_line
                    # Loop until find a next line node that already has parcel
                    while current.next_node:
                        if current.next_node.parcel:
                            break
                        current = current.next_node
                    current.parcel = self.parcel
                    self.parcel = None
                    return StepResult(
                        Picker.DROP_OFF_REWARD,
                        [warehouse.Movement(current.parcel, current.world_pos)],
                    )
        return StepResult(0.0, None)

    @property
    @override
    def state(self):
        if self.parcel:
            has_parcel = 1.0 if self.parcel.is_requested else 0.5
        else:
            has_parcel = 0.0
        return np.array(
            [
                self.pos.x / self.map_size.x,
                self.pos.y / self.map_size.y,
                has_parcel,
            ],
            dtype=np.float32,
        )

    def __has_requested(self, from_line: map.LineNode):
        current = from_line
        while True:
            if current.parcel and current.parcel.is_requested:
                return True
            if not current.previous_node:
                break
            current = current.previous_node
        return False
