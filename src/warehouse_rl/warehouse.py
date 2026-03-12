from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from csv import Error
from dataclasses import dataclass
from typing import Any, override

import numpy as np
import numpy.typing as npt
import pygame
from gymnasium.core import Env
from pygame.color import Color
from pygame.math import Vector2

from warehouse_rl import sprites
from warehouse_rl.enums import Action, Direction, RenderMode

pygame.init()

NODE_SIZE = Vector2(50, 30)
FRAME_PER_STEP = 5


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
    __map_size: Vector2
    __n_line_nodes: int
    ray_nodes: dict[str, RayNode]
    line_nodes: dict[str, LineNode]
    scene: pygame.Surface | None

    def __init__(
        self,
        n_rows: int,
        n_columns: int,
        n_subrows: int,
        n_lines: int,
        n_rays: int,
        render_mode: RenderMode = RenderMode.NoRender,
    ):
        assert n_rays in (1, 2)
        self.__create_rays(n_rows, n_columns, n_subrows, n_lines, n_rays)
        self.__create_lines(n_rows, n_columns, n_subrows, n_lines, n_rays)
        self.__map_size = Vector2(
            (n_lines + 1) * n_columns + 1,
            n_rows * n_subrows + 4 + n_rays * (n_rows - 1),
        )
        self.__n_line_nodes = n_rows * n_columns * n_subrows * n_lines
        match render_mode:
            case RenderMode.Human:
                screen_size = Vector2(
                    (n_lines + 1) * n_columns + 1,
                    n_rows * n_subrows + 4 + n_rays * (n_rows - 1) + 1,
                )
                self.scene = pygame.Surface(screen_size.elementwise() * NODE_SIZE)
            case RenderMode.NoRender:
                self.scene = None
            case _:
                raise ValueError(f"Invalid render_mode value: {render_mode}.")
        self.__draw()

    @property
    def map_size(self):
        return self.__map_size

    @property
    def n_line_nodes(self):
        return self.__n_line_nodes

    def __create_lines(
        self, n_rows: int, n_columns: int, n_subrows: int, n_lines: int, n_rays: int
    ):
        # add line nodes
        for row in range(n_rows):
            for column in range(n_columns):
                for innner_row in range(n_subrows):
                    for innner_column in range(n_lines):
                        line_node = LineNode(
                            column * (n_lines + 1) + innner_column + 1,
                            row * (n_subrows + n_rays) + innner_row + 2,
                        )
                        self.line_nodes[line_node.id] = line_node
        # add palletize node
        palletizer = LineNode(1, -1, True)
        self.line_nodes[palletizer.id] = palletizer
        # add link from ray to line
        for ray_node in self.ray_nodes.values():
            for line_node in self.line_nodes.values():
                if (line_node.x == ray_node.x) and (line_node.y - ray_node.y == -1):
                    ray_node.from_line = line_node
                if (line_node.x == ray_node.x) and (line_node.y - ray_node.y == 1):
                    ray_node.to_line = line_node
        # add link between line nodes
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
        # add horizontal edges
        for row in range(n_rows + 1):
            # double ray in first and last rays
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
        # add vertical ray
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
        if self.scene:
            pygame.draw.line(self.scene, color, start, end, width)
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
            pygame.draw.polygon(self.scene, color, (end, left, right))

    def __draw(self):
        if self.scene:
            self.scene.fill((255, 255, 255))
            for ray_node in self.ray_nodes.values():
                ray_node.draw(self.scene)
            for line_node in self.line_nodes.values():
                line_node.draw(self.scene)
            for ray_node in self.ray_nodes.values():
                for neighbor in ray_node.neighbors:
                    self.__draw_arrow(
                        Color(168, 177, 179),
                        ray_node.world_pos,
                        neighbor.world_pos,
                    )


@dataclass
class Obsevation:
    shuttle_state: npt.NDArray[np.float32]
    mask: npt.NDArray[np.uint8]


class Warehouse(Env[Obsevation, int]):
    __n_steps: int
    n_parcels: int
    max_step: int
    map: WarehouseMap
    shuttle: sprites.Shuttle
    render_mode: RenderMode
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
        render_mode: RenderMode = RenderMode.NoRender,
    ) -> None:
        super().__init__()
        self.__n_steps = 0
        self.n_parcels = 0
        n_rays: int = 2 if is_double_line else 1
        self.map = WarehouseMap(
            n_rows, n_columns, n_subrows, n_lines, n_rays, render_mode
        )
        self.shuttle = sprites.Shuttle(
            self.map.ray_nodes["2.0"], self.map.map_size, render_mode
        )
        _ = sprites.Parcel(self.map.line_nodes[f"1.{-1}"], render_mode)
        self.max_step = max_step
        self.render_mode = render_mode  # type: ignore
        match render_mode:
            case RenderMode.Human:
                if self.map.scene:
                    self.screen = pygame.display.set_mode(self.map.scene.get_size())
                self.clock = pygame.time.Clock()
                pygame.display.set_caption("WAREHOUSE")
            case RenderMode.NoRender:
                self.screen = None
                self.clock = None
            case _:
                raise ValueError(f"Invalid render_mode value: {render_mode}.")

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.__n_steps = 0
        self.n_parcel = 0
        self.shuttle.reset(self.map.ray_nodes["2.0"])
        for line_node in self.map.line_nodes.values():
            line_node.parcel = None
        _ = sprites.Parcel(self.map.line_nodes[f"1.{-1}"], self.render_mode)
        self.render()
        observation = Obsevation(self.shuttle.observation, self.shuttle.mask)
        info: dict[str, Any] = {}
        return observation, info

    @override
    def step(self, action: int):
        termination = self.n_parcels == self.map.n_line_nodes
        truncation = self.__n_steps == self.max_step
        if termination or truncation:
            raise Error("The environment has ended.")
        reward = self.shuttle.step(Action(action), self)
        self.__n_steps += 1
        observation = Obsevation(self.shuttle.observation, self.shuttle.mask)
        termination = self.n_parcels == self.map.n_line_nodes
        truncation = self.__n_steps == self.max_step
        info: dict[str, Any] = {}
        return observation, reward, termination, truncation, info

    @override
    def render(self):
        if self.screen and self.map.scene and self.clock:
            self.screen.blit(self.map.scene, (0, 0))
            self.shuttle.draw(self.screen)
            for line_node in self.map.line_nodes.values():
                if line_node.parcel:
                    line_node.parcel.draw(self.screen)
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.update()


if __name__ == "__main__":
    env = Warehouse(2, 2, 2, 2, True, 700, RenderMode.Human)
    running = True
    obs, _ = env.reset()
    for i in range(700):
        if not running:
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
        action_mask = obs.mask
        legal_actions = [i + 1 for i, v in enumerate(action_mask) if v]
        action = random.choice(legal_actions)
        obs, reward, termination, truncation, _ = env.step(action)
        print(
            f"In step {i}: reward {reward} termination {termination} truncation {truncation}"
        )

    # env.render()
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_r:
    #                 env.reset()
    #             if event.key == pygame.K_w:
    #                 env.step(1)
    #             if event.key == pygame.K_s:
    #                 env.step(2)
    #             if event.key == pygame.K_a:
    #                 env.step(3)
    #             if event.key == pygame.K_d:
    #                 env.step(4)
    pygame.quit()
