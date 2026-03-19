from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, override

import cv2
import numpy as np
import numpy.typing as npt
import pygame
from gymnasium import spaces
from gymnasium.core import Env
from pygame.math import Vector2

from warehouse_rl import map, sprites
from warehouse_rl.enums import (
    STATE_SIZE,
    Action,
    ObsMode,
    RenderMode,
)

SPEED = 250


@dataclass
class Observation:
    obs: npt.NDArray[np.float32]
    mask: npt.NDArray[np.uint8]


@dataclass
class Movement:
    sprite: sprites.Sprite
    # The target cooridnates must not change during movement,
    # Otherwise, the the trajectory is curve but not straight line
    target: Vector2


class Warehouse(Env[Observation, npt.NDArray[np.integer]]):
    step_counter: int
    parcel_counter: int
    n_steps: int
    n_shuttles: int
    map: map.WarehouseMap
    shuttles: list[sprites.Shuttle]
    obs_mode: ObsMode
    screen: pygame.Surface | None
    clock: pygame.time.Clock | None
    __recording: bool
    __writer: cv2.VideoWriter | None
    metadata: dict[str, Any] = {
        "render_modes": ["human"],
        "name": "warehouse",
        "is_parallelizable": True,
        "render_fps": 23,
    }

    def __init__(
        self,
        n_rows: int,
        n_columns: int,
        n_subrows: int,
        n_lines: int,
        is_double_line: bool,
        n_steps: int,
        n_shuttles: int,
        render_mode: RenderMode = RenderMode.Null,
        observation_mode: ObsMode = ObsMode.Flatten,
        recording: bool = False,
    ) -> None:
        super().__init__()
        self.step_counter = 0
        self.parcel_counter = 0
        self.n_steps = n_steps
        self.n_shuttles = n_shuttles
        n_rays: int = 2 if is_double_line else 1
        self.map = map.WarehouseMap(n_rows, n_columns, n_subrows, n_lines, n_rays)
        self.shuttles = []
        for ray_node in random.sample(list(self.map.ray_nodes.values()), n_shuttles):
            self.shuttles.append(sprites.Loader(ray_node, self.map.map_size))
        self.action_space = spaces.MultiDiscrete(np.full(n_shuttles, 4))
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
        self.recording = recording

    @property
    def recording(self):
        return self.__recording

    @recording.setter
    def recording(self, recording: bool):
        self.__recording = recording
        if self.__recording:
            fourcc = int(cv2.VideoWriter_fourcc(*"mp4v"))  # type: ignore
            self.__writer = cv2.VideoWriter(
                "warehouse.mp4", fourcc, 10, self.map.image.get_size()
            )
        else:
            self.__writer = None

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
        self.parcel_counter = 0
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
        if (
            self.parcel_counter == self.map.n_line_nodes
            or self.step_counter == self.n_steps
        ):
            raise ValueError(
                "The environment has ended. You have to reset it before step it further."
            )
        reward_a = [0.0] * self.n_shuttles
        # TODO: Moving shuttle by deterministic order is not close to parallelism.
        # Should we make movement order random?
        shuttle_movements: list[Movement] = []
        for i, shuttle in enumerate(self.shuttles):
            result = shuttle.step(Action(action[i]))
            if result.movements:
                reward_a[i] = result.reward
                shuttle_movements.extend(result.movements)
        self.__simulate_movement(shuttle_movements)
        # TODO: If we want parcel movement is parallel with shuttle movement,
        # we have to add new action. Pick or drop parcel right away after shuttle movement
        # and simulate all is not right because the target which sprite move to cann't move
        # during its movement.
        parcel_movements: list[Movement] = []
        for i, shuttle in enumerate(self.shuttles):
            result = shuttle.pick_up()
            if result.movements:
                reward_a[i] = result.reward
                parcel_movements.extend(result.movements)
            result = shuttle.drop_off()
            if result.movements:
                self.parcel_counter += 1
                reward_a[i] = result.reward
                parcel_movements.extend(result.movements)
        self.__simulate_movement(parcel_movements)

        self.step_counter += 1
        obs = self.__make_observation()
        termination = self.parcel_counter == self.map.n_line_nodes
        truncation = self.step_counter == self.n_steps
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

    @override
    def close(self) -> None:
        if self.__writer:
            self.__writer.release()

    def __make_observation(self):
        match self.obs_mode:
            case ObsMode.Flatten:
                line_nodes_states: list[float] = []
                for line_node in self.map.line_nodes.values():
                    if not line_node.is_depalletized and not line_node.is_palletized:
                        if line_node.parcel:
                            line_nodes_states.append(1.0)
                        else:
                            line_nodes_states.append(0.0)
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
            case _:
                raise ValueError(
                    f"Invalid render_mode value: {self.__observation_mode}."
                )
        mask_a_ac = np.vstack([shuttle.mask for shuttle in self.shuttles])
        return Observation(obs, mask_a_ac)

    def __simulate_movement(self, movements: list[Movement]):
        # Simulate if rendering to screen or recording
        if (self.screen and self.clock) or self.__recording:
            not_reaches = [True] * len(movements)
            while any(not_reaches):
                for i, movement in enumerate(movements):
                    if not_reaches[i]:
                        direction = movement.target - movement.sprite.world_pos
                        distance = direction.length()
                        dt = (
                            self.clock.tick(self.metadata["render_fps"]) / 1000
                            if self.clock
                            else 1 / self.metadata["render_fps"]
                        )
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
                surf = (
                    self.screen
                    if self.screen
                    else pygame.Surface(self.map.image.get_size())
                )
                self.__render_to_surface(surf)
                self.__write_frame(surf)
                if self.screen and self.clock:
                    pygame.display.update()
        elif self.obs_mode == ObsMode.ResizedWindow:
            for movement in movements:
                movement.sprite.world_pos = movement.target

    def __create_obs_img(self):
        surf = pygame.Surface(self.map.image.get_size())
        self.__render_to_surface(surf)
        scaled_screen = pygame.transform.smoothscale(surf, STATE_SIZE)
        # Transpose to torch convention dimension order (C, H, W)
        arr_c_h_w = np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(2, 0, 1)
        )
        # Compute per-channel min and max
        min_val_c_h_w = arr_c_h_w.min(axis=(1, 2), keepdims=True)
        max_val_c_h_w = arr_c_h_w.max(axis=(1, 2), keepdims=True)
        # Min-max normalize per channel to [0,1]
        return (arr_c_h_w - min_val_c_h_w) / (max_val_c_h_w - min_val_c_h_w)

    def __write_frame(self, surface: pygame.Surface):
        if self.__recording and self.__writer:
            scaled_screen = pygame.transform.smoothscale(surface, surface.get_size())
            # Transpose to dimension order (W, H, C)
            frame = np.transpose(
                np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
            )
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.__writer.write(frame)

    def __render_to_surface(self, surface: pygame.Surface):
        surface.blit(self.map.image, (0, 0))
        for shuttle in self.shuttles:
            shuttle.draw(surface)
        for line_node in self.map.line_nodes.values():
            if line_node.parcel:
                line_node.parcel.draw(surface)


if __name__ == "__main__":
    pygame.init()
    env = Warehouse(
        2,
        2,
        3,
        3,
        True,
        n_steps=200,
        n_shuttles=3,
        render_mode=RenderMode.Human,
        recording=False
    )
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
        mask_a_ac = obs.mask
        action_a: list[int] = []
        for mask_ac in mask_a_ac:
            legal_action_ac = [i for i, v in enumerate(mask_ac) if v]
            if len(legal_action_ac) != 0:
                action_a.append(random.choice(legal_action_ac))
            else:
                action_a.append(1)
        next_obs, reward_a, termination, truncation, info = env.step(np.array(action_a))
        # print(
        #     f"In step {env.n_steps}: observation {obs.obs} action {action_a} reward {reward_a}"
        # )
        obs = next_obs
        done = termination or truncation
    env.close()
    pygame.quit()
