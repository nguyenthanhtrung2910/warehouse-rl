from __future__ import annotations

import abc
import dataclasses
from itertools import islice
import typing

import numpy as np
import numpy.typing as npt
import pygame
import pygame.math

import warehouse_rl.enums
import warehouse_rl.map
import warehouse_rl.warehouse


@dataclasses.dataclass
class StepResult:
    reward: float
    movements: list[warehouse_rl.warehouse.Movement] | None


class Sprite(abc.ABC):
    image: pygame.Surface
    rect: pygame.Rect

    def __init__(self):
        self.image = pygame.Surface(warehouse_rl.enums.NODE_SIZE, pygame.SRCALPHA)
        self.rect = self.image.get_rect()

    @property
    def world_pos(self):
        return pygame.math.Vector2(self.rect.center)

    @world_pos.setter
    @abc.abstractmethod
    def world_pos(self, world_pos: pygame.math.Vector2):
        pass

    @abc.abstractmethod
    def world_translate(self, bias: pygame.math.Vector2):
        pass

    @abc.abstractmethod
    def draw(self, screen: pygame.Surface):
        pass


class Parcel(Sprite):
    __is_requested: bool

    def __init__(self, pos: warehouse_rl.map.LineNode, is_requested: bool = False):
        super().__init__()
        pos.parcel = self
        color = (158, 52, 235) if is_requested else (0, 255, 0)
        pygame.draw.circle(
            self.image,
            color,
            warehouse_rl.enums.NODE_SIZE / 2,
            min(warehouse_rl.enums.NODE_SIZE) / 4,
        )
        self.rect.center = pos.world_pos  # pyright: ignore[reportAttributeAccessIssue]
        self.__is_requested = is_requested

    @Sprite.world_pos.setter
    def world_pos(self, world_pos: pygame.math.Vector2):
        self.rect.center = world_pos  # pyright: ignore[reportAttributeAccessIssue]

    @typing.override
    def world_translate(self, bias: pygame.math.Vector2):
        self.rect.move_ip(bias)

    @typing.override
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
            warehouse_rl.enums.NODE_SIZE / 2,
            min(warehouse_rl.enums.NODE_SIZE) / 4,
        )


class Shuttle(Sprite):
    DEFAULT_REWARD = -0.1
    map_size: pygame.math.Vector2
    pos: warehouse_rl.map.RayNode
    parcel: Parcel | None

    def __init__(
        self,
        pos: warehouse_rl.map.RayNode,
        map_size: pygame.math.Vector2,
    ):
        super().__init__()
        self.map_size = map_size
        self.pos = pos
        self.pos.robot = self
        self.parcel = None
        self.rect.center = pos.world_pos  # pyright: ignore[reportAttributeAccessIssue]

    @abc.abstractmethod
    def pick_up(self) -> StepResult:
        pass

    @abc.abstractmethod
    def drop_off(self) -> StepResult:
        pass

    @Sprite.world_pos.setter
    def world_pos(self, world_pos: pygame.math.Vector2):
        self.rect.center = world_pos  # pyright: ignore[reportAttributeAccessIssue]
        if self.parcel:
            self.parcel.world_pos = world_pos

    @typing.override
    def world_translate(self, bias: pygame.math.Vector2):
        self.rect.move_ip(bias)
        if self.parcel:
            self.parcel.world_translate(bias)

    @typing.override
    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)
        if self.parcel:
            self.parcel.draw(screen)

    @property
    def mask(self):
        return np.array(
            [self._is_legal_move(action) for action in warehouse_rl.enums.Action],
            dtype=np.uint8,
        )

    @property
    @abc.abstractmethod
    def state(self) -> npt.NDArray[np.float64]:
        pass

    def reset(self, pos: warehouse_rl.map.RayNode):
        self.pos.robot = None
        self.pos = pos
        self.pos.robot = self
        self.parcel = None
        self.world_pos = pos.world_pos

    def step(self, action: warehouse_rl.enums.Action):
        # Check if action is legal
        # Actually, action from agent always is legal because of action mask,
        # we check for case that all actions are illegal
        is_action_legal = self._is_legal_move(action)
        if not is_action_legal:
            # If no action is legal, do nothing
            return StepResult(0.0, None)
        match action:
            case warehouse_rl.enums.Action.Null:
                return StepResult(self.DEFAULT_REWARD, None)
            case warehouse_rl.enums.Action.Up:
                self.__move_up()
            case warehouse_rl.enums.Action.Down:
                self.__move_down()
            case warehouse_rl.enums.Action.Left:
                self.__move_left()
            case warehouse_rl.enums.Action.Right:
                self.__move_right()
            case _:
                raise ValueError(f"Invalid action value {action}.")
        return StepResult(
            Shuttle.DEFAULT_REWARD,
            [warehouse_rl.warehouse.Movement(self, self.pos.world_pos)],
        )

    def _is_legal_move(self, act: warehouse_rl.enums.Action):
        match act:
            case warehouse_rl.enums.Action.Null:
                return True
            case warehouse_rl.enums.Action.Up:
                if not self.pos.up:
                    return False
                if self.pos.up.robot:
                    return False
            case warehouse_rl.enums.Action.Down:
                if not self.pos.down:
                    return False
                if self.pos.down.robot:
                    return False
            case warehouse_rl.enums.Action.Left:
                if not self.pos.left:
                    return False
                if self.pos.left.robot:
                    return False
            case warehouse_rl.enums.Action.Right:
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
        pos: warehouse_rl.map.RayNode,
        map_size: pygame.math.Vector2,
    ):
        super().__init__(pos, map_size)
        pygame.draw.circle(
            self.image,
            (255, 0, 0),
            warehouse_rl.enums.NODE_SIZE / 2,
            min(warehouse_rl.enums.NODE_SIZE) / 2,
        )

    @typing.override
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
                Loader.PICKUP_REWARD,
                [warehouse_rl.warehouse.Movement(self.parcel, self.world_pos)],
            )
        return StepResult(0.0, None)

    @typing.override
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
                [warehouse_rl.warehouse.Movement(current.parcel, current.world_pos)],
            )
        return StepResult(0.0, None)

    @property
    @typing.override
    def state(self):
        has_parcel = 1.0 if self.parcel else 0.0
        state: list[float] = [
            self.pos.x / self.map_size.x,
            self.pos.y / self.map_size.y,
            has_parcel,
        ]
        state.extend(
            [float(self._is_legal_move(action)) for action in islice(warehouse_rl.enums.Action, 1 , None)]
        )
        return np.array(state, dtype=np.float64)


class Picker(Shuttle):
    PICK_UP_REWARD = 1.0
    PICK_UP_REQ_REWARD = 2.0
    DROP_OFF_REWARD = 5.0
    DROP_OFF_REQ_REWARD = 10.0

    def __init__(
        self,
        pos: warehouse_rl.map.RayNode,
        map_size: pygame.math.Vector2,
    ):
        super().__init__(pos, map_size)
        pygame.draw.circle(
            self.image,
            (235, 119, 52),
            warehouse_rl.enums.NODE_SIZE / 2,
            min(warehouse_rl.enums.NODE_SIZE) / 2,
        )

    @typing.override
    def pick_up(self):
        if (
            self.pos.from_line
            and not self.pos.from_line.is_depalletized
            and self.pos.from_line.parcel
            and not self.parcel
            and self.__has_requested(self.pos.from_line)
        ):
            movements: list[warehouse_rl.warehouse.Movement] = []
            self.parcel = self.pos.from_line.parcel
            self.pos.from_line.parcel = None
            movements.append(
                warehouse_rl.warehouse.Movement(self.parcel, self.world_pos)
            )
            current = self.pos.from_line
            # Loop until no previous node or previous node has no parcel
            while current.previous_node and current.previous_node.parcel:
                current.parcel = current.previous_node.parcel
                current.previous_node.parcel = None
                movements.append(
                    warehouse_rl.warehouse.Movement(current.parcel, current.world_pos)
                )
                current = current.previous_node
            if self.parcel.is_requested:
                return StepResult(Picker.PICK_UP_REQ_REWARD, movements)
            else:
                return StepResult(Picker.PICK_UP_REWARD, movements)
        return StepResult(0.0, None)

    @typing.override
    def drop_off(self):
        if self.pos.to_line and not self.pos.to_line.parcel and self.parcel:
            if self.pos.to_line.is_palletized:
                if self.parcel.is_requested:
                    self.pos.to_line.parcel = self.parcel
                    self.parcel = None
                    return StepResult(
                        Picker.DROP_OFF_REQ_REWARD,
                        [
                            warehouse_rl.warehouse.Movement(
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
                        [
                            warehouse_rl.warehouse.Movement(
                                current.parcel, current.world_pos
                            )
                        ],
                    )
        return StepResult(0.0, None)

    @property
    @typing.override
    def state(self):
        if self.parcel:
            has_parcel = 1.0 if self.parcel.is_requested else 0.5
        else:
            has_parcel = 0.0
        state: list[float] = [
            self.pos.x / self.map_size.x,
            self.pos.y / self.map_size.y,
            has_parcel,
        ]
        state.extend(
            [float(self._is_legal_move(action)) for action in islice(warehouse_rl.enums.Action, 1 , None)]
        )
        return np.array(state, dtype=np.float64)

    def __has_requested(self, from_line: warehouse_rl.map.LineNode):
        current = from_line
        while True:
            if current.parcel and current.parcel.is_requested:
                return True
            if not current.previous_node:
                break
            current = current.previous_node
        return False
