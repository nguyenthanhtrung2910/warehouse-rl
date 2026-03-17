from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

import numpy as np
import numpy.typing as npt
import pygame
from gymnasium import spaces
from gymnasium.core import Env
from pygame.color import Color
from pygame.math import Vector2

from warehouse_rl import sprites
from warehouse_rl.enums import (
    NODE_SIZE,
    STATE_SIZE,
    Action,
    Direction,
    ObsMode,
    RenderMode,
)

pygame.init()

SPEED = 300
DEFAULT_REWARD = -0.1
PICKUP_REWARD = 1
DROPOFF_REWARD = 5


class Node(ABC):
    __x: int
    __y: int
    __id: str
    __world_pos: Vector2

    def __init__(self, x: int, y: int):
        self.__x = x
        self.__y = y
        self.__id = f"{self.__x}.{self.__y}"
        self.__world_pos = (
            Vector2(self.__x + 0.5, self.__y + 1.5).elementwise() * NODE_SIZE
        )

    def __eq__(self, other: object):
        if not isinstance(other, Node):
            return NotImplemented
        return self.__x == other.x and self.__y == other.y

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def id(self):
        return self.__id

    @property
    def world_pos(self):
        return self.__world_pos

    @abstractmethod
    def draw(self, surface: pygame.Surface):
        pass


class RayNode(Node):
    isRobotSpawn: bool
    up: RayNode | None
    down: RayNode | None
    left: RayNode | None
    right: RayNode | None
    from_line: LineNode | None
    to_line: LineNode | None
    robot: sprites.Shuttle | None

    def __init__(
        self,
        x: int,
        y: int,
        isRobotSpawn: bool = False,
        up: RayNode | None = None,
        down: RayNode | None = None,
        left: RayNode | None = None,
        right: RayNode | None = None,
        from_line: LineNode | None = None,
        to_line: LineNode | None = None,
        robot: sprites.Shuttle | None = None,
    ):
        super().__init__(x, y)
        self.isRobotSpawn = isRobotSpawn
        self.up = up
        self.down = down
        self.left = left
        self.right = right
        self.from_line = from_line
        self.to_line = to_line
        self.robot = robot

    @property
    def neighbors(self):
        return [node for node in [self.up, self.down, self.left, self.right] if node]

    @override
    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(surface, (0, 0, 255), self.world_pos, min(NODE_SIZE) / 4)
        if self.from_line:
            pygame.draw.line(
                surface, (30, 0, 30), self.world_pos, self.from_line.world_pos, 1
            )
        if self.to_line:
            pygame.draw.line(
                surface, (30, 30, 0), self.world_pos, self.to_line.world_pos, 1
            )


class LineNode(Node):
    isPalletize: bool
    next_node: LineNode | None
    parcel: sprites.Parcel | None

    def __init__(
        self,
        x: int,
        y: int,
        isPalletize: bool = False,
        next_node: LineNode | None = None,
        parcel: sprites.Parcel | None = None,
    ):
        super().__init__(x, y)
        self.isPalletize = isPalletize
        self.next_node = next_node
        self.parcel = parcel

    @override
    def draw(self, surface: pygame.Surface):
        radius = min(NODE_SIZE) / 3 if self.isPalletize else min(NODE_SIZE) / 4
        pygame.draw.circle(surface, (255, 255, 0), self.world_pos, radius)
        if self.next_node:
            pygame.draw.line(
                surface, (30, 30, 0), self.world_pos, self.next_node.world_pos, 1
            )


class WarehouseMap:
    map_size: Vector2
    n_line_nodes: int
    ray_nodes: dict[str, RayNode]
    line_nodes: dict[str, LineNode]
    image: pygame.Surface

    def __init__(
        self,
        n_rows: int,
        n_columns: int,
        n_subrows: int,
        n_lines: int,
        n_rays: int,
    ):
        assert n_rays in (1, 2)
        self.__create_rays(n_rows, n_columns, n_subrows, n_lines, n_rays)
        self.__create_lines(n_rows, n_columns, n_subrows, n_lines, n_rays)
        self.map_size = Vector2(
            (n_lines + 1) * n_columns + 1,
            n_rows * n_subrows + 4 + n_rays * (n_rows - 1),
        )
        self.n_line_nodes = n_rows * n_columns * n_subrows * n_lines
        image_size = Vector2(
            (n_lines + 1) * n_columns + 1,
            n_rows * n_subrows + 4 + n_rays * (n_rows - 1) + 1,
        )
        self.image = pygame.Surface(image_size.elementwise() * NODE_SIZE)
        self.__draw()

    def __create_lines(
        self, n_rows: int, n_columns: int, n_subrows: int, n_lines: int, n_rays: int
    ):
        # Add line nodes
        for row in range(n_rows):
            for column in range(n_columns):
                for innner_row in range(n_subrows):
                    for innner_column in range(n_lines):
                        line_node = LineNode(
                            column * (n_lines + 1) + innner_column + 1,
                            row * (n_subrows + n_rays) + innner_row + 2,
                        )
                        self.line_nodes[line_node.id] = line_node
        # Add palletize node
        palletizer = LineNode(1, -1, True)
        self.line_nodes[palletizer.id] = palletizer
        # Add link from ray to line
        for ray_node in self.ray_nodes.values():
            for line_node in self.line_nodes.values():
                if (line_node.x == ray_node.x) and (line_node.y - ray_node.y == -1):
                    ray_node.from_line = line_node
                if (line_node.x == ray_node.x) and (line_node.y - ray_node.y == 1):
                    ray_node.to_line = line_node
        # Add link between line nodes
        for line_node1 in self.line_nodes.values():
            for line_node2 in self.line_nodes.values():
                if (line_node1.x == line_node2.x) and (
                    line_node1.y - line_node2.y == 1
                ):
                    line_node2.next_node = line_node1

    def __create_rays(
        self, n_rows: int, n_columns: int, n_subrows: int, n_lines: int, n_rays: int
    ):
        self.ray_nodes = {}
        self.line_nodes = {}
        direction = False
        line_begin_end: list[tuple[RayNode, RayNode]] = []
        # Add horizontal edges
        for row in range(n_rows + 1):
            # Double ray in first and last rays
            if row == 0:
                for pair in range(0, 2):
                    line_begin_end.append(
                        self.__create_horizontal_ray(
                            (n_lines + 1) * n_columns + 1,
                            pair,
                            direction,
                        )
                    )
                    direction = not direction
            elif row != n_rows:
                for pair in range(0, n_rays):
                    line_begin_end.append(
                        self.__create_horizontal_ray(
                            (n_lines + 1) * n_columns + 1,
                            (n_subrows + n_rays) * row - n_rays + pair + 2,
                            direction,
                        )
                    )
                    direction = not direction
            else:
                for pair in range(0, 2):
                    line_begin_end.append(
                        self.__create_horizontal_ray(
                            (n_lines + 1) * n_columns + 1,
                            (n_subrows + n_rays) * row - n_rays + pair + 2,
                            direction,
                        )
                    )
                    direction = not direction
        # Add vertical ray
        for i in range(len(line_begin_end) - 1):
            self.__create_ray_edge(
                line_begin_end[i][0], line_begin_end[i + 1][0], Direction.Down
            )
        for i in range(len(line_begin_end) - 1, 0, -1):
            self.__create_ray_edge(
                line_begin_end[i][1], line_begin_end[i - 1][1], Direction.Up
            )

    def __create_ray_edge(self, n1: RayNode, n2: RayNode, direction: Direction):
        self.ray_nodes.setdefault(n1.id, n1)
        self.ray_nodes.setdefault(n2.id, n2)
        n_1 = self.ray_nodes[n1.id]
        n_2 = self.ray_nodes[n2.id]
        match direction:
            case Direction.Up:
                n_1.up = n_2
            case Direction.Down:
                n_1.down = n_2
            case Direction.Left:
                n_1.left = n_2
            case Direction.Right:
                n_1.right = n_2
            case _:
                raise ValueError(f"Invalid direction value {direction}")

    def __create_horizontal_ray(self, n_nodes: int, y: int, positive_direction: bool):
        if positive_direction:
            for x in range(n_nodes - 1):
                self.__create_ray_edge(
                    RayNode(x, y), RayNode(x + 1, y), Direction.Right
                )
        else:
            for x in range(n_nodes - 1, 0, -1):
                self.__create_ray_edge(RayNode(x, y), RayNode(x - 1, y), Direction.Left)
        return self.ray_nodes[f"0.{y}"], self.ray_nodes[f"{n_nodes - 1}.{y}"]

    def __draw_arrow(
        self,
        color: Color,
        start: Vector2,
        end: Vector2,
        width: int = 2,
        arrow_size: float = 10,
    ):
        pygame.draw.line(self.image, color, start, end, width)
        dx = end.x - start.x
        dy = end.y - start.y
        angle = math.atan2(dy, dx)
        left = (
            end.x - arrow_size * math.cos(angle - math.pi / 6),
            end.y - arrow_size * math.sin(angle - math.pi / 6),
        )
        right = (
            end.x - arrow_size * math.cos(angle + math.pi / 6),
            end.y - arrow_size * math.sin(angle + math.pi / 6),
        )
        pygame.draw.polygon(self.image, color, (end, left, right))

    def __draw(self):
        self.image.fill((255, 255, 255))
        for ray_node in self.ray_nodes.values():
            ray_node.draw(self.image)
        for line_node in self.line_nodes.values():
            line_node.draw(self.image)
        for ray_node in self.ray_nodes.values():
            for neighbor in ray_node.neighbors:
                self.__draw_arrow(
                    Color(168, 177, 179),
                    ray_node.world_pos,
                    neighbor.world_pos,
                )


@dataclass
class Observation:
    obs: npt.NDArray[np.float32]
    mask: npt.NDArray[np.uint8]


@dataclass
class Movement:
    sprite: sprites.Sprite
    target: Vector2


class Warehouse(Env[Observation, npt.NDArray[np.integer]]):
    n_steps: int
    n_parcels: int
    n_shuttles: int
    max_step: int
    map: WarehouseMap
    shuttles: list[sprites.Shuttle]
    obs_mode: ObsMode
    screen: pygame.Surface | None
    clock: pygame.time.Clock | None
    metadata: dict[str, Any] = {
        "render_modes": ["human"],
        "name": "warehouse",
        "is_parallelizable": True,
        "render_fps": 20,
    }

    def __init__(
        self,
        n_rows: int,
        n_columns: int,
        n_subrows: int,
        n_lines: int,
        is_double_line: bool,
        max_step: int,
        n_shuttles: int,
        render_mode: RenderMode = RenderMode.Null,
        observation_mode: ObsMode = ObsMode.Flatten,
    ) -> None:
        super().__init__()
        self.n_steps = 0
        self.n_parcels = 0
        self.n_shuttles = n_shuttles
        n_rays: int = 2 if is_double_line else 1
        self.max_step = max_step
        self.map = WarehouseMap(n_rows, n_columns, n_subrows, n_lines, n_rays)
        self.shuttles = []
        for ray_node in random.sample(list(self.map.ray_nodes.values()), n_shuttles):
            self.shuttles.append(sprites.Shuttle(ray_node, self.map.map_size))
        _ = sprites.Parcel(self.map.line_nodes[f"1.{-1}"])
        self.action_space = spaces.MultiDiscrete(np.full(n_shuttles, 4))
        self.rend_mode = render_mode
        self.obs_mode = observation_mode
        match render_mode:
            case RenderMode.Null:
                self.screen = None
                self.clock = None
            case RenderMode.Human:
                self.screen = pygame.display.set_mode(self.map.image.get_size())
                self.clock = pygame.time.Clock()
                pygame.display.set_caption("WAREHOUSE")
            case _:
                raise ValueError(f"Invalid render_mode value: {render_mode}.")

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed:
            random.seed(seed)
        self.n_steps = 0
        self.n_parcels = 0
        for shuttle, ray_node in zip(
            self.shuttles,
            random.sample(list(self.map.ray_nodes.values()), self.n_shuttles),
        ):
            shuttle.reset(ray_node)
        for line_node in self.map.line_nodes.values():
            line_node.parcel = None
        _ = sprites.Parcel(self.map.line_nodes[f"1.{-1}"])
        self.render()
        obs = self.__make_observation()
        info: dict[str, Any] = {}
        return obs, info

    @override
    def step(self, action: npt.NDArray[np.integer]):
        if self.n_parcels == self.map.n_line_nodes or self.n_steps == self.max_step:
            raise ValueError(
                "The environment has ended. You have to reset it before step it further."
            )
        reward_a = [0.0] * self.n_shuttles

        shuttle_movements: list[Movement] = []
        for i, shuttle in enumerate(self.shuttles):
            is_moved = shuttle.step(Action(action[i]))
            if is_moved:
                reward_a[i] = DEFAULT_REWARD
                if self.rend_mode == RenderMode.Human:
                    shuttle_movements.append(Movement(shuttle, shuttle.pos.world_pos))
                if self.rend_mode == RenderMode.Null and (
                    self.obs_mode == ObsMode.ResizedWindow
                    or self.obs_mode == ObsMode.FullWindow
                ):
                    shuttle.world_pos = shuttle.pos.world_pos
        self.__simulate_movement(shuttle_movements)

        parcel_movements: list[Movement] = []
        for i, shuttle in enumerate(self.shuttles):
            parcel = shuttle.pick_up()
            if parcel:
                reward_a[i] = PICKUP_REWARD
                if self.rend_mode == RenderMode.Human:
                    parcel_movements.append(Movement(parcel, shuttle.world_pos))
                if self.rend_mode == RenderMode.Null and (
                    self.obs_mode == ObsMode.ResizedWindow
                    or self.obs_mode == ObsMode.FullWindow
                ):
                    parcel.world_pos = shuttle.world_pos
        self.__simulate_movement(parcel_movements)

        parcel_movements: list[Movement] = []
        for i, shuttle in enumerate(self.shuttles):
            to_line, parcel = shuttle.drop_off()
            if to_line and parcel:
                reward_a[i] = DROPOFF_REWARD
                self.n_parcels += 1
                if self.rend_mode == RenderMode.Human:
                    parcel_movements.append(Movement(parcel, to_line.world_pos))
                if self.rend_mode == RenderMode.Null and (
                    self.obs_mode == ObsMode.ResizedWindow
                    or self.obs_mode == ObsMode.FullWindow
                ):
                    parcel.world_pos = to_line.world_pos
        self.__simulate_movement(parcel_movements)

        self.n_steps += 1
        obs = self.__make_observation()
        termination = self.n_parcels == self.map.n_line_nodes
        truncation = self.n_steps == self.max_step
        info: dict[str, Any] = {}
        return (
            obs,
            np.array(reward_a),
            termination,
            truncation,
            info,
        )

    @override
    def render(self):
        if self.screen and self.clock:
            self.clock.tick(self.metadata["render_fps"])
            self.__render_to_surface(self.screen)
            pygame.display.update()

    def __make_observation(self):
        match self.obs_mode:
            case ObsMode.Flatten:
                line_nodes_states = [
                    1 if line_node.parcel else 0
                    for line_node in self.map.line_nodes.values()
                    if not line_node.isPalletize
                ]
                # obs <==> obs_a_o
                # TODO: Should we add at least state of arounding shuttles to each 
                # shuttle's observation? For centralized training?
                obs = np.vstack(
                    [
                        np.hstack((shuttle.state, np.array(line_nodes_states)))
                        for shuttle in self.shuttles
                    ]
                )
            case ObsMode.ResizedWindow:
                # TODO: obs <==> obs_a_c_h_w
                obs = self.__create_obs_img()
            case ObsMode.FullWindow:
                # obs <==> obs_w_h_c
                obs = self.__create_frame()
            case _:
                raise ValueError(
                    f"Invalid render_mode value: {self.__observation_mode}."
                )
        mask_a_ac = np.vstack([shuttle.mask for shuttle in self.shuttles])
        return Observation(obs, mask_a_ac)

    def __simulate_movement(self, movements: list[Movement]):
        if self.screen and self.clock:
            not_reaches = [True] * len(movements)
            while any(not_reaches):
                for i, movement in enumerate(movements):
                    if not_reaches[i]:
                        direction = movement.target - movement.sprite.world_pos
                        distance = direction.length()
                        dt = self.clock.tick(self.metadata["render_fps"]) / 1000
                        step = SPEED * dt
                        if distance <= step or distance == 0:
                            # Correct the final shuttle positions
                            movement.sprite.world_pos = movement.target
                            not_reaches[i] = False
                        else:
                            # Translate by small distance
                            movement.sprite.world_translate(
                                direction.normalize() * step
                            )
                self.__render_to_surface(self.screen)
                pygame.display.update()

    def __create_obs_img(self):
        surf = pygame.Surface(self.map.image.get_size())
        self.__render_to_surface(surf)
        scaled_screen = pygame.transform.smoothscale(surf, STATE_SIZE)
        # Transpose to torch convention dimension order (C, H, W)
        arr_chw = np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(2, 0, 1)
        )
        # Compute per-channel min and max
        min_val = arr_chw.min(axis=(1, 2), keepdims=True)
        max_val = arr_chw.max(axis=(1, 2), keepdims=True)
        # Min-max normalize per channel to [0,1]
        return (arr_chw - min_val) / (max_val - min_val)

    def __create_frame(self):
        surf = pygame.Surface(self.map.image.get_size())
        self.__render_to_surface(surf)
        scaled_screen = pygame.transform.smoothscale(surf, surf.get_size())
        # Transpose to dimension order (W, H, C)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def __render_to_surface(self, surface: pygame.Surface):
        surface.blit(self.map.image, (0, 0))
        for shuttle in self.shuttles:
            shuttle.draw(surface)
        for line_node in self.map.line_nodes.values():
            if line_node.parcel:
                line_node.parcel.draw(surface)


if __name__ == "__main__":
    env = Warehouse(2, 2, 2, 2, True, 200, 3, RenderMode.Human, ObsMode.Flatten)
    obs, info = env.reset()
    done = False
    running = True
    while not done and running:
        if not running:
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
        mask_a_ac = obs.mask
        action_a: list[int] = []
        for mask_ac in mask_a_ac:
            legal_action_ac = [i for i, v in enumerate(mask_ac) if v]
            if len(legal_action_ac) != 0:
                action_a.append(random.choice(legal_action_ac))
            else:
                action_a.append(1)
        next_obs, reward_a, termination, truncation, info = env.step(np.array(action_a))
        print(
            f"In step {env.n_steps}: observation {obs.obs} action {action_a} reward {reward_a}"
        )
        obs = next_obs
        done = termination or truncation

    pygame.quit()
