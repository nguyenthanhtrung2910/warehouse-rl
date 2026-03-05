# from pettingzoo import ParallelEnv
# import json
# import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx


@dataclass
class Node:
    id: str
    x: int
    y: int
    isRobotSpawn: bool
    robot: object = None

    # Hash based only on the node ID
    def __hash__(self):
        return hash(self.id)

    # Equality based on ID
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id


@dataclass(frozen=True)
class LineNode:
    id: str
    x: int
    y: int
    isPaletize: bool


class Warehouse:
    __agents: list[str]
    __graph: nx.DiGraph

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
        self.__agents = [f"a{i}" for i in range(n_robots)]
        self.__init_graph(n_rows, n_columns, n_lines, n_subrows, is_double_line)

    def __add_edge(self, n1: Node, n2: Node, direction: bool):
        if direction:
            self.__graph.add_edge(n1, n2)
        else:
            self.__graph.add_edge(n2, n1)

    def __init_graph(
        self,
        n_rows: int,
        n_columns: int,
        n_lines: int,
        n_subrows: int,
        is_double_line: bool,
    ):
        self.__graph = nx.DiGraph()
        direction: bool = False
        n_rays: int = 2 if is_double_line else 1
        for column in range((n_lines + 1) * n_columns):
            for row in range(n_rows + 1):
                # double ray in first and last rays
                if row == n_rows:
                    for pair in range(0, 2):
                        self.__add_edge(
                            Node(
                                f"{column}.{(n_subrows + n_rays) * row - n_rays + pair + 2}",
                                column,
                                (n_subrows + n_rays) * row - n_rays + pair + 2,
                                False,
                            ),
                            Node(
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
                        self.__add_edge(
                            Node(
                                f"{column}.{pair}",
                                column,
                                pair,
                                False,
                            ),
                            Node(
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
                        self.__add_edge(
                            Node(
                                f"{column}.{(n_subrows + n_rays) * row - n_rays + pair + 2}",
                                column,
                                (n_subrows + n_rays) * row - n_rays + pair + 2,
                                False,
                            ),
                            Node(
                                f"{column + 1}.{(n_subrows + n_rays) * row - n_rays + pair + 2}",
                                column + 1,
                                (n_subrows + n_rays) * row - n_rays + pair + 2,
                                False,
                            ),
                            direction,
                        )
                        direction = not direction
        for row in range(n_rows):
            for column in range(n_columns):
                for innner_row in range(n_subrows):
                    for innner_column in range(n_lines):
                        self.__graph.add_node(
                            LineNode(
                                f"{column * (n_lines + 1) + innner_column + 1}.{row * (n_subrows + n_rays) + innner_row + 2}",
                                column * (n_lines + 1) + innner_column + 1,
                                row * (n_subrows + n_rays) + innner_row + 2,
                                False,
                            )
                        )
        for n1 in self.__graph.nodes:
            for n2 in self.__graph.nodes:
                if (n2.y - n1.y == 1) or (n2.y - n1.y == (n_subrows + 1)):
                    if n1.x == 0 and n2.x == 0:
                        self.__add_edge(n1, n2, True)
                    if (
                        n1.x == (n_lines + 1) * n_columns
                        and n2.x == (n_lines + 1) * n_columns
                    ):
                        self.__add_edge(n1, n2, False)

    def draw(self):
        pos: dict[Node | LineNode, tuple[int, int]] = {
            node: (node.x, node.y) for node in self.__graph.nodes
        }
        plt.figure(figsize=(20, 16), dpi=100)
        node_colors = []
        for node in self.__graph.nodes:
            if type(node) is LineNode:
                node_colors.append("yellow")
            else:
                node_colors.append("lightblue")
        nx.draw(
            G=self.__graph,
            pos=pos,
            node_color=node_colors,
            node_size=200,
            arrowsize=20,
        )
        plt.savefig("graph.png")

    # def reset(self, seed: int | None = None, options: dict | None = None):
    #     return super().reset(seed, options)

    # def step(self, actions: dict[str, int]):
    #     return super().step(actions)

    # def render(self):
    #     return super().render()

    # def close(self):
    #     return super().close()


if __name__ == "__main__":
    env = Warehouse(2, 2, 3, 5, 5, True)
    env.draw()
