from __future__ import annotations

import random
from typing import Any, override

import numpy as np
import numpy.typing as npt
import pygame

from warehouse_rl import map, sprites, warehouse
from warehouse_rl.enums import (
    Action,
    ObsMode,
    RenderMode,
)


class WarehouseB(warehouse.Warehouse):
    n_parcels: int
    n_requested_parcels: int
    taken_parcel_counter: int

    def __init__(
        self,
        n_rows: int,
        n_columns: int,
        n_subrows: int,
        n_lines: int,
        is_double_line: bool,
        n_steps: int,
        n_shuttles: int,
        n_parcels: int,
        n_requested: int,
        render_mode: RenderMode = RenderMode.Null,
        observation_mode: ObsMode = ObsMode.Flatten,
    ) -> None:
        super().__init__(
            n_rows,
            n_columns,
            n_subrows,
            n_lines,
            is_double_line,
            n_steps,
            n_shuttles,
            render_mode,
            observation_mode,
        )
        self.taken_parcel_counter = 0
        self.n_parcels = n_parcels
        self.n_requested_parcels = n_requested
        for ray_node in self.map.ray_nodes.values():
            if ray_node.robot:
                ray_node.robot = None
        self.shuttles.clear()
        for ray_node in random.sample(list(self.map.ray_nodes.values()), n_shuttles):
            self.shuttles.append(sprites.Picker(ray_node, self.map.map_size))

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed:
            random.seed(seed)
        self.step_counter = 0
        self.taken_parcel_counter = 0
        for shuttle, ray_node in zip(
            self.shuttles,
            random.sample(list(self.map.ray_nodes.values()), self.n_shuttles),
        ):
            shuttle.reset(ray_node)
        for line_node in self.map.line_nodes.values():
            line_node.parcel = None
        center_line_nodes = [
            line_node
            for line_node in self.map.line_nodes.values()
            if not line_node.is_depalletized and not line_node.is_palletized
        ]
        filled_line_nodes: list[map.LineNode] = random.sample(
            center_line_nodes, self.n_parcels
        )
        for line_node in filled_line_nodes:
            _ = sprites.Parcel(line_node, False)
        for line_node in random.sample(filled_line_nodes, self.n_requested_parcels):
            if line_node.parcel:
                line_node.parcel.is_requested = True
        for ray_node in self.map.ray_nodes.values():
            if ray_node.from_line and not ray_node.from_line.is_depalletized:
                self.__downfall_parcel(ray_node.from_line)
        self.render()
        obs = self._make_observation()
        info: dict[str, Any] = {}
        return obs, info

    @override
    def step(self, action: npt.NDArray[np.integer]):
        if (
            self.taken_parcel_counter == self.n_requested_parcels
            or self.step_counter == self.n_steps
        ):
            raise ValueError(
                "The environment has ended. You have to reset it before step it further."
            )
        reward_a = [0.0] * self.n_shuttles
        # TODO: Moving shuttle by deterministic order is not close to parallelism.
        # Should we make movement order random?
        shuttle_movements: list[warehouse.Movement] = []
        for i, shuttle in enumerate(self.shuttles):
            result = shuttle.step(Action(action[i]))
            if result.movements:
                reward_a[i] = result.reward
                shuttle_movements.extend(result.movements)
        self._simulate_movement(shuttle_movements)
        # TODO: If we want parcel movement is parallel with shuttle movement,
        # we have to add new action. Pick or drop parcel right away after shuttle movement
        # and simulate all is not right because the target which sprite move to cann't move
        # during its movement.
        parcel_movements: list[warehouse.Movement] = []
        for i, shuttle in enumerate(self.shuttles):
            result = shuttle.pick_up()
            if result.movements:
                reward_a[i] = result.reward
                parcel_movements.extend(result.movements)
            result = shuttle.drop_off()
            if result.movements:
                reward_a[i] = result.reward
                parcel_movements.extend(result.movements)
        self._simulate_movement(parcel_movements)
        if self.map.palletized_node.parcel:
            self.taken_parcel_counter += 1
            self.map.palletized_node.parcel = None
        self.step_counter += 1
        obs = self._make_observation()
        termination = self.taken_parcel_counter == self.n_requested_parcels
        truncation = self.step_counter == self.n_steps
        info: dict[str, Any] = {}
        return (
            obs,
            np.array(reward_a),
            termination,
            truncation,
            info,
        )

    def __downfall_parcel(self, from_line: map.LineNode):
        filled_line_nodes: list[map.LineNode] = []
        current = from_line
        while True:
            if current.parcel:
                filled_line_nodes.append(current)
            if not current.previous_node:
                break
            current = current.previous_node
        current = from_line
        for filled_node in filled_line_nodes:
            if filled_node is not current:
                current.parcel = filled_node.parcel
                current.parcel.world_pos = current.world_pos  # pyright: ignore[reportOptionalMemberAccess]
                filled_node.parcel = None
            if current.previous_node:
                current = current.previous_node
            else:
                break


if __name__ == "__main__":
    pygame.init()
    env = WarehouseB(2, 2, 2, 2, True, 500, 1, 12, 2, RenderMode.Human)
    obs, info = env.reset()
    done = False
    running = True
    while not done and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
        env.render()
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
        # env.render()
        # while running:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             running = False
        #         if event.type == pygame.KEYDOWN:
        #             if event.key == pygame.K_r:
        #                 env.reset()
        #             if event.key == pygame.K_w:
        #                 env.step(np.array([0]))
        #             if event.key == pygame.K_s:
        #                 env.step(np.array([1]))
        #             if event.key == pygame.K_a:
        #                 env.step(np.array([2]))
        #             if event.key == pygame.K_d:
        #                 env.step(np.array([3]))
    pygame.quit()
