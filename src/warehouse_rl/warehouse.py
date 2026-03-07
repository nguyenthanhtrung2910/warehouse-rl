# from pettingzoo import ParallelEnv
from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass
from typing import Any

import pygame

from warehouse_rl import sprites

pygame.init()

NODE_SIZE = (46, 30)
FRAME_PER_STEP = 5


class Direction(Enum):
    Up = 1
    Down = 2
    Left = 3
    Right = 4


def draw_arrow(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    start: tuple[float, float],
    end: tuple[float, float],
    width: int = 2,
    arrow_size: float = 10,
):
    pygame.draw.line(surface, color, start, end, width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    left = (
        end[0] - arrow_size * math.cos(angle - math.pi / 6),
        end[1] - arrow_size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - arrow_size * math.cos(angle + math.pi / 6),
        end[1] - arrow_size * math.sin(angle + math.pi / 6),
    )
    pygame.draw.polygon(surface, color, [end, left, right])


@dataclass
class RayNode:
    x: int
    y: int
    isRobotSpawn: bool
    up: RayNode | None = None
    down: RayNode | None = None
    left: RayNode | None = None
    right: RayNode | None = None
    robot: sprites.Shuttle | None = None
    from_line: LineNode | None = None
    to_line: LineNode | None = None

    @property
    def id(self):
        return f"{self.x}.{self.y}"

    @property
    def adjacent(self):
        return [node for node in [self.up, self.down, self.left, self.right] if node]

    @property
    def world_pos(self):
        return (self.x + 0.5) * NODE_SIZE[0], (self.y + 0.5) * NODE_SIZE[1]

    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(
            surface,
            color=(0, 0, 255),
            center=self.world_pos,
            radius=min(NODE_SIZE) / 4,
        )
        if self.from_line:
            pygame.draw.line(
                surface, (30, 0, 30), self.world_pos, self.from_line.world_pos, 1
            )
        if self.to_line:
            pygame.draw.line(
                surface, (30, 30, 0), self.world_pos, self.to_line.world_pos, 1
            )
        # rect = pygame.Rect(0, 0, NODE_SIZE[0], NODE_SIZE[1])
        # rect.center = self.world_pos
        # pygame.draw.rect(surface, (0, 0, 0), rect, 1)


@dataclass
class LineNode:
    x: int
    y: int
    isPaletize: bool

    @property
    def id(self):
        return f"{self.x}.{self.y}"

    @property
    def world_pos(self):
        return (self.x + 0.5) * NODE_SIZE[0], (self.y + 0.5) * NODE_SIZE[1]

    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(
            surface,
            color=(255, 255, 0),
            center=self.world_pos,
            radius=min(NODE_SIZE) / 4,
        )


class WarehouseMap:
    ray_nodes: dict[str, RayNode]
    line_nodes: dict[str, LineNode]

    def __init__(
        self,
        n_rows: int,
        n_columns: int,
        n_lines: int,
        n_subrows: int,
        n_rays: int,
    ):
        assert n_rays in (1, 2)
        self.ray_nodes = {}
        self.line_nodes = {}
        direction: bool = False
        line_begin_end = []
        # add horizontal edges
        for row in range(n_rows + 1):
            # double ray in first and last rays
            if row == 0:
                for pair in range(0, 2):
                    line_begin_end.append(
                        self.__add_horizontal_ray(
                            (n_lines + 1) * n_columns + 1,
                            pair,
                            direction,
                        )
                    )
                    direction = not direction
            elif row != n_rows:
                for pair in range(0, n_rays):
                    line_begin_end.append(
                        self.__add_horizontal_ray(
                            (n_lines + 1) * n_columns + 1,
                            (n_subrows + n_rays) * row - n_rays + pair + 2,
                            direction,
                        )
                    )
                    direction = not direction
            else:
                for pair in range(0, 2):
                    line_begin_end.append(
                        self.__add_horizontal_ray(
                            (n_lines + 1) * n_columns + 1,
                            (n_subrows + n_rays) * row - n_rays + pair + 2,
                            direction,
                        )
                    )
                    direction = not direction

        # add line nodes
        for row in range(n_rows):
            for column in range(n_columns):
                for innner_row in range(n_subrows):
                    for innner_column in range(n_lines):
                        line_node = LineNode(
                            column * (n_lines + 1) + innner_column + 1,
                            row * (n_subrows + n_rays) + innner_row + 2,
                            False,
                        )
                        self.line_nodes[line_node.id] = line_node

        # add vertical ray
        for i in range(len(line_begin_end) - 1):
            self.__add_ray_edge(
                line_begin_end[i][0], line_begin_end[i + 1][0], Direction.Down
            )
        for i in range(len(line_begin_end) - 1, 0, -1):
            self.__add_ray_edge(
                line_begin_end[i][1], line_begin_end[i - 1][1], Direction.Up
            )

        # add link from ray to line
        for ray_node in self.ray_nodes.values():
            for line_node in self.line_nodes.values():
                if (line_node.x == ray_node.x) and (line_node.y - ray_node.y == -1):
                    ray_node.from_line = line_node
                if (line_node.x == ray_node.x) and (line_node.y - ray_node.y == 1):
                    ray_node.to_line = line_node

    def __add_ray_edge(self, n1: RayNode, n2: RayNode, direction: Direction):
        self.ray_nodes.setdefault(n1.id, n1)
        self.ray_nodes.setdefault(n2.id, n2)
        n_1 = self.ray_nodes[n1.id]
        n_2 = self.ray_nodes[n2.id]
        if direction == Direction.Up:
            n_1.up = n_2
        elif direction == Direction.Down:
            n_1.down = n_2
        elif direction == Direction.Left:
            n_1.left = n_2
        elif direction == Direction.Right:
            n_1.right = n_2
        else:
            raise ValueError("No available direction.")

    def __add_horizontal_ray(self, n_nodes, y, positive_direction: bool):
        if positive_direction:
            for x in range(n_nodes - 1):
                self.__add_ray_edge(
                    RayNode(x, y, False), RayNode(x + 1, y, False), Direction.Right
                )
        else:
            for x in range(n_nodes - 1, 0, -1):
                self.__add_ray_edge(
                    RayNode(x, y, False), RayNode(x - 1, y, False), Direction.Left
                )
        return self.ray_nodes[f"0.{y}"], self.ray_nodes[f"{n_nodes - 1}.{y}"]


class Warehouse:
    agents: list[str]
    robot: sprites.Shuttle
    robot_sprites: pygame.sprite.Group
    map: WarehouseMap
    screen_size: tuple[int, int]
    background: pygame.Surface
    clock: pygame.time.Clock
    metadata: dict[str, Any] = {
        "render_modes": ["human"],
        "name": "warehouse",
        "is_parallelizable": True,
        "render_fps": 20,
    }
    screen: pygame.Surface | None

    def __init__(
        self,
        n_robots: int,
        n_rows: int,
        n_columns: int,
        n_lines: int,
        n_subrows: int,
        is_double_line: bool,
    ) -> None:
        # super().__init__()
        self.agents = [f"a.{i}" for i in range(n_robots)]
        n_rays: int = 2 if is_double_line else 1
        self.map = WarehouseMap(n_rows, n_columns, n_lines, n_subrows, n_rays)
        self.screen_size = (
            (n_lines + 1) * n_columns + 1,
            n_rows * n_subrows + 4 + n_rays * (n_rows - 1),
        )
        self.robot = sprites.Shuttle(self.map.ray_nodes["0.0"])
        self.robot_sprites = pygame.sprite.Group()
        self.robot_sprites.add(self.robot)
        self.robot_sprites.add(sprites.Shuttle(self.map.ray_nodes["4.1"]))
        # draw a background
        self.background = pygame.Surface(
            (NODE_SIZE[0] * self.screen_size[0], NODE_SIZE[1] * self.screen_size[1])
        )
        self.background.fill((255, 255, 255))
        # draw graph
        for ray_node in self.map.ray_nodes.values():
            ray_node.draw(self.background)
        for line_node in self.map.line_nodes.values():
            line_node.draw(self.background)
        for node in self.map.ray_nodes.values():
            for adjacent_node in node.adjacent:
                draw_arrow(
                    self.background,
                    (168, 177, 179),
                    node.world_pos,
                    adjacent_node.world_pos,
                )
        self.clock = pygame.time.Clock()
        self.screen = None

    # def reset(self, seed: int | None = None, options: dict | None = None):
    #     return super().reset(seed, options)

    def step(self, action: sprites.Action):
        self.robot.step(action)
        # for smooth movement
        for i in range(1, FRAME_PER_STEP + 1):
            diff = tuple(
                a - b
                for a, b in zip(self.robot.next_rect.center, self.robot.rect.center)
            )
            self.robot.rect.center = tuple(
                a + i / FRAME_PER_STEP * b for a, b in zip(self.robot.rect.center, diff)
            )
            self.render()

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode(self.background.get_size())
            pygame.display.set_caption("WAREHOUSE")
        self.screen.blit(self.background, (0, 0))
        self.robot_sprites.draw(self.screen)
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.update()

    # def close(self):
    #     return super().close()


if __name__ == "__main__":
    env = Warehouse(2, 3, 3, 5, 5, False)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    env.step(sprites.Action.UP)
                if event.key == pygame.K_s:
                    env.step(sprites.Action.DOWN)
                if event.key == pygame.K_a:
                    env.step(sprites.Action.LEFT)
                if event.key == pygame.K_d:
                    env.step(sprites.Action.RIGHT)
        env.render()
    pygame.quit()
