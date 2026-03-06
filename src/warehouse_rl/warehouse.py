# from pettingzoo import ParallelEnv
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import networkx as nx
import pygame
from warehouse_rl import sprites
import math

pygame.init()

NODE_SIZE = (30, 30)


def draw_arrow(surface, color, start, end, width=2, arrow_size=10):
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


@dataclass(frozen=False)
class RayNode:
    id: str
    x: int
    y: int
    isRobotSpawn: bool
    up: RayNode | None = None
    down: RayNode | None = None
    left: RayNode | None = None
    right: RayNode | None = None
    robot: sprites.Shuttle | None = None

    # hash based only on the node ID
    def __hash__(self):
        return hash(self.id)

    # equality based on ID
    def __eq__(self, other: RayNode):
        if not isinstance(other, RayNode):
            return False
        return self.id == other.id

    @property
    def world_pos(self):
        return (self.x + 0.5) * NODE_SIZE[0], (self.y + 0.5) * NODE_SIZE[1]

    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(
            surface,
            color=(0, 0, 255),
            center=self.world_pos,
            radius=NODE_SIZE[0] / 4,
        )


@dataclass(frozen=True)
class LineNode:
    id: str
    x: int
    y: int
    isPaletize: bool

    @property
    def world_pos(self):
        return (self.x + 0.5) * NODE_SIZE[0], (self.y + 0.5) * NODE_SIZE[1]

    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(
            surface,
            color=(255, 255, 0),
            center=self.world_pos,
            radius=NODE_SIZE[0] / 4,
        )


class Warehouse:
    agents: list[str]
    graph: nx.DiGraph
    size: tuple[int, int]
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
        self.agents = [f"a{i}" for i in range(n_robots)]
        n_rays: int = 2 if is_double_line else 1
        self.__init_graph(n_rows, n_columns, n_lines, n_subrows, n_rays)
        self.size = (
            (n_lines + 1) * n_columns + 1,
            n_rows * n_subrows + 4 + n_rays * (n_rows - 1),
        )
        # draw a background
        self.background = pygame.Surface(
            (NODE_SIZE[0] * self.size[0], NODE_SIZE[1] * self.size[1])
        )
        self.background.fill((255, 255, 255))
        # draw graph
        for node in self.graph.nodes:
            node.draw(self.background)
        for node in self.graph.nodes:
            for next in self.graph.successors(node):
                draw_arrow(self.background, (0, 0, 0), node.world_pos, next.world_pos)
        self.clock = pygame.time.Clock()
        self.screen = None

    def __add_edge(self, n1: RayNode, n2: RayNode, direction: bool):
        if direction:
            self.graph.add_edge(n1, n2)
        else:
            self.graph.add_edge(n2, n1)

    def __init_graph(
        self,
        n_rows: int,
        n_columns: int,
        n_lines: int,
        n_subrows: int,
        n_rays: int,
    ):
        self.graph = nx.DiGraph()
        direction: bool = False
        # add horizontal edges
        for row in range(n_rows + 1):
            # double ray in first and last rays
            if row == n_rows:
                for pair in range(0, 2):
                    for column in range((n_lines + 1) * n_columns):
                        self.__add_edge(
                            RayNode(
                                f"{column}.{(n_subrows + n_rays) * row - n_rays + pair + 2}",
                                column,
                                (n_subrows + n_rays) * row - n_rays + pair + 2,
                                False,
                            ),
                            RayNode(
                                f"{column + 1}.{(n_subrows + n_rays) * row - n_rays + pair + 2}",
                                column + 1,
                                (n_subrows + n_rays) * row - n_rays + pair + 2,
                                False,
                            ),
                            direction,
                        )
                    direction = not direction
            elif row == 0:
                for pair in range(0, 2):
                    for column in range((n_lines + 1) * n_columns):
                        self.__add_edge(
                            RayNode(
                                f"{column}.{pair}",
                                column,
                                pair,
                                False,
                            ),
                            RayNode(
                                f"{column + 1}.{pair}",
                                column + 1,
                                pair,
                                False,
                            ),
                            direction,
                        )
                    direction = not direction
            # else add number rays based on is_double_line
            else:
                for pair in range(0, n_rays):
                    for column in range((n_lines + 1) * n_columns):
                        self.__add_edge(
                            RayNode(
                                f"{column}.{(n_subrows + n_rays) * row - n_rays + pair + 2}",
                                column,
                                (n_subrows + n_rays) * row - n_rays + pair + 2,
                                False,
                            ),
                            RayNode(
                                f"{column + 1}.{(n_subrows + n_rays) * row - n_rays + pair + 2}",
                                column + 1,
                                (n_subrows + n_rays) * row - n_rays + pair + 2,
                                False,
                            ),
                            direction,
                        )
                    direction = not direction
        # add line nodes
        for row in range(n_rows):
            for column in range(n_columns):
                for innner_row in range(n_subrows):
                    for innner_column in range(n_lines):
                        self.graph.add_node(
                            LineNode(
                                f"{column * (n_lines + 1) + innner_column + 1}.{row * (n_subrows + n_rays) + innner_row + 2}",
                                column * (n_lines + 1) + innner_column + 1,
                                row * (n_subrows + n_rays) + innner_row + 2,
                                False,
                            )
                        )
        # add vertical edges
        for n1 in self.graph.nodes:
            for n2 in self.graph.nodes:
                if (n2.y - n1.y == 1) or (n2.y - n1.y == (n_subrows + 1)):
                    if n1.x == 0 and n2.x == 0:
                        self.__add_edge(n1, n2, True)
                    if (
                        n1.x == (n_lines + 1) * n_columns
                        and n2.x == (n_lines + 1) * n_columns
                    ):
                        self.__add_edge(n1, n2, False)
                if (
                    (n2.y - n1.y == 1)
                    and (n1.x == n2.x)
                    and (
                        (type(n2) is LineNode and type(n1) is RayNode)
                        or (type(n1) is LineNode and type(n2) is RayNode)
                    )
                ):
                    self.__add_edge(n2, n1, False)

        for node in self.graph.nodes:
            if type(node) is RayNode:
                for next in self.graph.successors(node):
                    if next.y - node.y == 1:
                        node.down = next
                    if next.y - node.y == -1:
                        node.up = next
                    if next.x - node.x == -1:
                        node.left = next
                    if next.x - node.x == 1:
                        node.right = next

    # def reset(self, seed: int | None = None, options: dict | None = None):
    #     return super().reset(seed, options)

    # def step(self, actions: dict[str, int]):
    #     return super().step(actions)

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode(self.background.get_size())
            pygame.display.set_caption("WAREHOUSE")
        self.screen.blit(self.background, (0, 0))
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.update()

    # def close(self):
    #     return super().close()


if __name__ == "__main__":
    env = Warehouse(2, 3, 3, 5, 5, False)
    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #     env.render()
    # pygame.quit()
