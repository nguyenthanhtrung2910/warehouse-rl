# from pettingzoo import ParallelEnv
import json
import os
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

    def __init__(self, n_robots: int) -> None:
        # super().__init__()
        self.__agents = [f"a{i}" for i in range(n_robots)]
        self.__init_graph()

    def __init_graph(self):
        self.__graph = nx.DiGraph()
        id_nodes = {}
        file_dir = os.path.dirname(os.path.abspath(__file__))
        with open(file_dir + "/../../wareshouse.json", "r") as file:
            warehouse_config = json.load(file)
        for node_data in warehouse_config["nodes"]:
            node = Node(**node_data)
            id_nodes[node.id] = node
        for edge_data in warehouse_config["edges"]:
            src_node = id_nodes[edge_data["source"]]
            tgt_node = id_nodes[edge_data["target"]]
            self.__graph.add_edge(src_node, tgt_node)
        for node_data in warehouse_config["lineNodes"]:
            self.__graph.add_node(LineNode(**node_data))

    def draw(self):
        pos: dict[Node | LineNode, tuple[int, int]] = {
            node: (node.x, node.y) for node in self.__graph.nodes
        }
        plt.figure(figsize=(20, 12), dpi=100)
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
            node_size=50,
            arrowsize=5,
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
    env = Warehouse(2)
    env.draw()
