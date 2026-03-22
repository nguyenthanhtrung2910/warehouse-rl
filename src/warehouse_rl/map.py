from __future__ import annotations

import abc
import math
import types
import typing

import pygame
import pygame.color
import pygame.math

import warehouse_rl.enums
import warehouse_rl.sprites


def draw_arrow(
    surface: pygame.Surface,
    color: pygame.color.Color,
    start: pygame.math.Vector2,
    end: pygame.math.Vector2,
    width: int = 2,
    arrow_size: float = 10,
) -> None:
    pygame.draw.line(surface, color, start, end, width)
    dx: float = end.x - start.x
    dy: float = end.y - start.y
    angle: float = math.atan2(dy, dx)
    left: tuple[float, float] = (
        end.x - arrow_size * math.cos(angle - math.pi / 6),
        end.y - arrow_size * math.sin(angle - math.pi / 6),
    )
    right: tuple[float, float] = (
        end.x - arrow_size * math.cos(angle + math.pi / 6),
        end.y - arrow_size * math.sin(angle + math.pi / 6),
    )
    pygame.draw.polygon(surface, color, (end, left, right))


class Node(abc.ABC):
    __x: int
    __y: int
    __id: str
    __world_pos: pygame.math.Vector2

    def __init__(self, x: int, y: int) -> None:
        self.__x = x
        self.__y = y
        self.__id = f"{self.__x}.{self.__y}"
        self.__world_pos = (
            pygame.math.Vector2(self.__x + 0.5, self.__y + 1.5).elementwise()
            * warehouse_rl.enums.NODE_SIZE
        )

    def __eq__(self, other: object) -> types.NotImplementedType | bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.__x == other.x and self.__y == other.y

    def __repr__(self) -> str:
        return f"{type(self)}({self.__x}, {self.y})"

    @property
    def x(self) -> int:
        return self.__x

    @property
    def y(self) -> int:
        return self.__y

    @property
    def id(self) -> str:
        return self.__id

    @property
    def world_pos(self) -> pygame.Vector2:
        return self.__world_pos

    @abc.abstractmethod
    def draw(self, surface: pygame.Surface) -> None:
        pass

    @abc.abstractmethod
    def draw_node_links(self, surface: pygame.Surface) -> None:
        pass


class RayNode(Node):
    isRobotSpawn: bool
    up: RayNode | None
    down: RayNode | None
    left: RayNode | None
    right: RayNode | None
    from_line: LineNode | None
    from_line: LineNode | None
    robot: warehouse_rl.sprites.Shuttle | None

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
        robot: warehouse_rl.sprites.Shuttle | None = None,
    ) -> None:
        super().__init__(x, y)
        self.isRobotSpawn = isRobotSpawn
        self.up = up
        self.down = down
        self.left = left
        self.right = right
        self.from_line = from_line
        self.to_line: LineNode | None = to_line
        self.robot = robot

    @property
    def neighbors(self) -> list[RayNode]:
        return [node for node in [self.up, self.down, self.left, self.right] if node]

    @typing.override
    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(
            surface, (0, 0, 255), self.world_pos, min(warehouse_rl.enums.NODE_SIZE) / 4
        )

    @typing.override
    def draw_node_links(self, surface: pygame.Surface) -> None:
        for neighbor in self.neighbors:
            if neighbor:
                draw_arrow(
                    surface,
                    pygame.color.Color(168, 177, 179),
                    self.world_pos,
                    neighbor.world_pos,
                    2,
                )
        if self.from_line:
            draw_arrow(
                surface,
                pygame.color.Color(168, 177, 179),
                self.from_line.world_pos,
                self.world_pos,
                2,
            )
        if self.to_line:
            draw_arrow(
                surface,
                pygame.color.Color(168, 177, 179),
                self.world_pos,
                self.to_line.world_pos,
                2,
            )


class LineNode(Node):
    is_depalletized: bool
    is_palletized: bool
    next_node: LineNode | None
    previous_node: LineNode | None
    parcel: warehouse_rl.sprites.Parcel | None

    def __init__(
        self,
        x: int,
        y: int,
        is_depalletized: bool = False,
        is_palletized: bool = False,
        next_node: LineNode | None = None,
        previous_node: LineNode | None = None,
        parcel: warehouse_rl.sprites.Parcel | None = None,
    ) -> None:
        super().__init__(x, y)
        self.is_depalletized = is_depalletized
        self.is_palletized = is_palletized
        self.next_node = next_node
        self.previous_node = previous_node
        self.parcel = parcel

    @typing.override
    def draw(self, surface: pygame.Surface) -> None:
        radius: float = (
            min(warehouse_rl.enums.NODE_SIZE) / 3
            if (self.is_depalletized or self.is_palletized)
            else min(warehouse_rl.enums.NODE_SIZE) / 4
        )
        pygame.draw.circle(surface, (255, 255, 0), self.world_pos, radius)

    @typing.override
    def draw_node_links(self, surface: pygame.Surface) -> None:
        if self.next_node:
            pygame.draw.line(
                surface,
                (36, 35, 20),
                self.world_pos + pygame.math.Vector2(3, 0),
                self.next_node.world_pos + pygame.math.Vector2(3, 0),
                2,
            )
        if self.previous_node:
            pygame.draw.line(
                surface,
                (36, 35, 20),
                self.world_pos + pygame.math.Vector2(-3, 0),
                self.previous_node.world_pos + pygame.math.Vector2(-3, 0),
                2,
            )


class WarehouseMap:
    map_size: pygame.math.Vector2
    n_line_nodes: int
    ray_nodes: dict[str, RayNode]
    line_nodes: dict[str, LineNode]
    image: pygame.Surface
    palletized_node: LineNode

    def __init__(
        self,
        n_rows: int,
        n_columns: int,
        n_subrows: int,
        n_lines: int,
        n_rays: int,
    ) -> None:
        assert n_rays in (1, 2)
        self.map_size = pygame.math.Vector2(
            (n_lines + 1) * n_columns + 1,
            n_rows * n_subrows + 4 + n_rays * (n_rows - 1),
        )
        self.__create_rays(n_rows, n_columns, n_subrows, n_lines, n_rays)
        self.__create_lines(n_rows, n_columns, n_subrows, n_lines, n_rays)
        self.n_line_nodes = n_rows * n_columns * n_subrows * n_lines
        image_size = pygame.math.Vector2(
            (n_lines + 1) * n_columns + 1,
            n_rows * n_subrows + 4 + n_rays * (n_rows - 1) + 2,
        )
        self.image = pygame.Surface(
            image_size.elementwise() * warehouse_rl.enums.NODE_SIZE
        )
        self.__draw()

    def __create_lines(
        self, n_rows: int, n_columns: int, n_subrows: int, n_lines: int, n_rays: int
    ) -> None:
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
        # Add depalletize node
        depalletizer = LineNode(1, -1, True, False)
        palletizer = LineNode(
            (n_lines + 1) * n_columns - 1,
            n_rows * n_subrows + 4 + n_rays * (n_rows - 1),
            False,
            True,
        )
        self.line_nodes[depalletizer.id] = depalletizer
        self.line_nodes[palletizer.id] = palletizer
        self.palletized_node = palletizer
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
                    line_node1.previous_node = line_node2

    def __create_rays(
        self, n_rows: int, n_columns: int, n_subrows: int, n_lines: int, n_rays: int
    ) -> None:
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
                    direction: bool = not direction
            elif row != n_rows:
                for pair in range(0, n_rays):
                    line_begin_end.append(
                        self.__create_horizontal_ray(
                            (n_lines + 1) * n_columns + 1,
                            (n_subrows + n_rays) * row - n_rays + pair + 2,
                            direction,
                        )
                    )
                    direction: bool = not direction
            else:
                for pair in range(0, 2):
                    line_begin_end.append(
                        self.__create_horizontal_ray(
                            (n_lines + 1) * n_columns + 1,
                            (n_subrows + n_rays) * row - n_rays + pair + 2,
                            direction,
                        )
                    )
                    direction: bool = not direction
        # Add vertical ray
        for i in range(len(line_begin_end) - 1):
            self.__create_ray_edge(
                line_begin_end[i][0],
                line_begin_end[i + 1][0],
                warehouse_rl.enums.Direction.Down,
            )
        for i in range(len(line_begin_end) - 1, 0, -1):
            self.__create_ray_edge(
                line_begin_end[i][1],
                line_begin_end[i - 1][1],
                warehouse_rl.enums.Direction.Up,
            )

    def __create_ray_edge(
        self, n1: RayNode, n2: RayNode, direction: warehouse_rl.enums.Direction
    ) -> None:
        self.ray_nodes.setdefault(n1.id, n1)
        self.ray_nodes.setdefault(n2.id, n2)
        n_1: RayNode = self.ray_nodes[n1.id]
        n_2: RayNode = self.ray_nodes[n2.id]
        match direction:
            case warehouse_rl.enums.Direction.Up:
                n_1.up = n_2
            case warehouse_rl.enums.Direction.Down:
                n_1.down = n_2
            case warehouse_rl.enums.Direction.Left:
                n_1.left = n_2
            case warehouse_rl.enums.Direction.Right:
                n_1.right = n_2
            case _:
                raise ValueError(f"Invalid direction value {direction}")

    def __create_horizontal_ray(
        self, n_nodes: int, y: int, positive_direction: bool
    ) -> tuple[RayNode, RayNode]:
        if positive_direction:
            for x in range(n_nodes - 1):
                self.__create_ray_edge(
                    RayNode(x, y), RayNode(x + 1, y), warehouse_rl.enums.Direction.Right
                )
        else:
            for x in range(n_nodes - 1, 0, -1):
                self.__create_ray_edge(
                    RayNode(x, y), RayNode(x - 1, y), warehouse_rl.enums.Direction.Left
                )
        return self.ray_nodes[f"0.{y}"], self.ray_nodes[f"{n_nodes - 1}.{y}"]

    def __draw(self) -> None:
        self.image.fill((255, 255, 255))
        for ray_node in self.ray_nodes.values():
            ray_node.draw(self.image)
        for line_node in self.line_nodes.values():
            line_node.draw(self.image)
        for line_node in self.line_nodes.values():
            line_node.draw_node_links(self.image)
        for ray_node in self.ray_nodes.values():
            ray_node.draw_node_links(self.image)
