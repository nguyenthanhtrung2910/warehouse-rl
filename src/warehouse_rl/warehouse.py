from __future__ import annotations

import dataclasses
import random
import typing

import cv2
import gymnasium
import gymnasium.core
import numpy as np
import pygame
import pygame.event
import pygame.math

import warehouse_rl.enums
import warehouse_rl.map
import warehouse_rl.sprites

SPEED = 250


@dataclasses.dataclass
class Observation:
    obs: np.ndarray[tuple[typing.Any, ...], np.dtype[np.floating]]
    mask: np.ndarray[tuple[typing.Any, ...], np.dtype[np.unsignedinteger]]


@dataclasses.dataclass
class Movement:
    sprite: warehouse_rl.sprites.Sprite
    # The target cooridnates must not change during movement,
    # Otherwise, the the trajectory is curve but not straight line
    target: pygame.math.Vector2


class Warehouse(
    gymnasium.core.Env[
        Observation, np.ndarray[tuple[typing.Any, ...], np.dtype[np.integer]]
    ]
):
    step_counter: int
    parcel_counter: int
    n_steps: int
    n_shuttles: int
    map: warehouse_rl.map.WarehouseMap
    shuttles: list[warehouse_rl.sprites.Shuttle]
    obs_mode: warehouse_rl.enums.ObsMode
    screen: pygame.Surface | None
    clock: pygame.time.Clock | None
    __recording: bool
    __writer: cv2.VideoWriter | None
    metadata: dict[str, typing.Any] = {
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
        render_mode: warehouse_rl.enums.RenderMode = warehouse_rl.enums.RenderMode.Null,
        observation_mode: warehouse_rl.enums.ObsMode = warehouse_rl.enums.ObsMode.Flatten,
        recording: bool = False,
    ) -> None:
        super().__init__()
        self.step_counter = 0
        self.parcel_counter = 0
        self.n_steps = n_steps
        self.n_shuttles = n_shuttles
        n_rays: int = 2 if is_double_line else 1
        self.map = warehouse_rl.map.WarehouseMap(
            n_rows, n_columns, n_subrows, n_lines, n_rays
        )
        self.shuttles = []
        for ray_node in random.sample(list(self.map.ray_nodes.values()), n_shuttles):
            self.shuttles.append(
                warehouse_rl.sprites.Loader(ray_node, self.map.map_size)
            )
        self.action_space = gymnasium.spaces.MultiDiscrete(np.full(n_shuttles, 4))
        self.obs_mode = observation_mode
        match render_mode:
            case warehouse_rl.enums.RenderMode.Null:
                self.screen = None
                self.clock = None
            case warehouse_rl.enums.RenderMode.Human:
                self.screen = pygame.display.set_mode(self.map.image.get_size())
                self.clock = pygame.time.Clock()
                pygame.display.set_caption("WAREHOUSE")
            case _:
                raise ValueError(f"Invalid render_mode value: {render_mode}.")
        self.recording = recording

    @property
    def recording(self) -> bool:
        return self.__recording

    @recording.setter
    def recording(self, recording: bool) -> None:
        self.__recording = recording
        if self.__recording:
            fourcc = int(cv2.VideoWriter_fourcc(*"mp4v"))  # type: ignore
            self.__writer = cv2.VideoWriter(
                "warehouse.mp4", fourcc, 10, self.map.image.get_size()
            )
        else:
            self.__writer = None

    @typing.override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,
    ) -> tuple[Observation, dict[str, typing.Any]]:
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
        _ = warehouse_rl.sprites.Parcel(self.map.line_nodes[f"1.{-1}"])
        self.render()
        obs: Observation = self.__make_observation()
        info: dict[str, typing.Any] = {}
        return obs, info

    @typing.override
    def step(
        self, action: np.ndarray[tuple[typing.Any, ...], np.dtype[np.integer]]
    ) -> tuple[
        Observation,
        np.ndarray[tuple[typing.Any, ...], np.dtype[np.floating]],
        bool,
        bool,
        dict[str, typing.Any],
    ]:
        if (
            self.parcel_counter == self.map.n_line_nodes
            or self.step_counter == self.n_steps
        ):
            raise ValueError(
                "The environment has ended. You have to reset it before step it further."
            )
        reward_a: list[float] = [0.0] * self.n_shuttles
        # TODO: Moving shuttle by deterministic order is not close to parallelism.
        # Should we make movement order random?
        shuttle_movements: list[Movement] = []
        for i, shuttle in enumerate(self.shuttles):
            result: warehouse_rl.sprites.StepResult = shuttle.step(
                warehouse_rl.enums.Action(action[i])
            )
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
            result: warehouse_rl.sprites.StepResult = shuttle.pick_up()
            if result.movements:
                reward_a[i] = result.reward
                parcel_movements.extend(result.movements)
            result: warehouse_rl.sprites.StepResult = shuttle.drop_off()
            if result.movements:
                self.parcel_counter += 1
                reward_a[i] = result.reward
                parcel_movements.extend(result.movements)
        self.__simulate_movement(parcel_movements)

        self.step_counter += 1
        obs: Observation = self.__make_observation()
        termination: bool = self.parcel_counter == self.map.n_line_nodes
        truncation: bool = self.step_counter == self.n_steps
        info: dict[str, typing.Any] = {}
        return (
            obs,
            np.array(reward_a),
            termination,
            truncation,
            info,
        )

    @typing.override
    def render(self) -> None:
        if self.screen and self.clock:
            self.clock.tick(self.metadata["render_fps"])
            self.__render_to_surface(self.screen)
            pygame.display.update()

    @typing.override
    def close(self) -> None:
        if self.__writer:
            self.__writer.release()

    def __make_observation(self) -> Observation:
        match self.obs_mode:
            case warehouse_rl.enums.ObsMode.Flatten:
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
                obs: np.ndarray[tuple[typing.Any, ...], np.dtype[np.floating]] = (
                    np.vstack(
                        [
                            np.hstack((shuttle.state, np.array(line_nodes_states)))
                            for shuttle in self.shuttles
                        ]
                    )
                )
            case warehouse_rl.enums.ObsMode.ResizedWindow:
                # TODO: obs <==> obs_a_c_h_w
                obs = self.__create_obs_img()
            case _:
                raise ValueError(
                    f"Invalid render_mode value: {self.__observation_mode}."
                )
        mask_a_ac: np.ndarray[tuple[typing.Any, ...], np.dtype[np.unsignedinteger]] = (
            np.vstack([shuttle.mask for shuttle in self.shuttles])
        )
        return Observation(obs, mask_a_ac)

    def __simulate_movement(self, movements: list[Movement]) -> None:
        # Simulate if rendering to screen or recording
        if (self.screen and self.clock) or self.__recording:
            not_reaches: list[bool] = [True] * len(movements)
            while any(not_reaches):
                for i, movement in enumerate(movements):
                    if not_reaches[i]:
                        direction: pygame.math.Vector2 = (
                            movement.target - movement.sprite.world_pos
                        )
                        distance: float = direction.length()
                        dt: float | typing.Any = (
                            self.clock.tick(self.metadata["render_fps"]) / 1000
                            if self.clock
                            else 1 / self.metadata["render_fps"]
                        )
                        step: float | typing.Any = SPEED * dt
                        if distance <= step or distance == 0:
                            # Correct the final shuttle positions
                            movement.sprite.world_pos = movement.target
                            not_reaches[i] = False
                        else:
                            # Translate by small distance
                            movement.sprite.world_translate(
                                direction.normalize() * step
                            )
                surf: pygame.Surface = (
                    self.screen
                    if self.screen
                    else pygame.Surface(self.map.image.get_size())
                )
                self.__render_to_surface(surf)
                self.__write_frame(surf)
                if self.screen and self.clock:
                    pygame.display.update()
        elif self.obs_mode == warehouse_rl.enums.ObsMode.ResizedWindow:
            for movement in movements:
                movement.sprite.world_pos = movement.target

    def __create_obs_img(self):
        surf = pygame.Surface(self.map.image.get_size())
        self.__render_to_surface(surf)
        scaled_screen: pygame.Surface = pygame.transform.smoothscale(
            surf, warehouse_rl.enums.STATE_SIZE
        )
        # Transpose to torch convention dimension order (C, H, W)
        arr_c_h_w: np.ndarray[tuple[typing.Any, ...], np.dtype[np.unsignedinteger]] = (
            np.transpose(
                np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(2, 0, 1)
            )
        )
        # Compute per-channel min and max
        min_val_c_h_w = arr_c_h_w.min(axis=(1, 2), keepdims=True)
        max_val_c_h_w = arr_c_h_w.max(axis=(1, 2), keepdims=True)
        # Min-max normalize per channel to [0,1]
        return (arr_c_h_w - min_val_c_h_w) / (max_val_c_h_w - min_val_c_h_w)

    def __write_frame(self, surface: pygame.Surface) -> None:
        if self.__recording and self.__writer:
            scaled_screen: pygame.Surface = pygame.transform.smoothscale(
                surface, surface.get_size()
            )
            # Transpose to dimension order (W, H, C)
            frame: np.ndarray[tuple[typing.Any, ...], np.dtype[np.unsignedinteger]] = (
                np.transpose(
                    np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
                )
            )
            # Convert RGB to BGR for OpenCV
            frame_bgr: (
                cv2.Mat
                | np.ndarray[
                    tuple[typing.Any, ...],
                    np.dtype[np.integer | np.floating],
                ]
            ) = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.__writer.write(frame_bgr)

    def __render_to_surface(self, surface: pygame.Surface) -> None:
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
        render_mode=warehouse_rl.enums.RenderMode.Human,
        recording=False,
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
            legal_action_ac: list[int] = [i for i, v in enumerate(mask_ac) if v]
            if len(legal_action_ac) != 0:
                action_a.append(random.choice(legal_action_ac))
            else:
                action_a.append(1)
        next_obs, reward_a, termination, truncation, info = env.step(np.array(action_a))
        # print(
        #     f"In step {env.n_steps}: observation {obs.obs} action {action_a} reward {reward_a}"
        # )
        obs: Observation = next_obs
        done: bool = termination or truncation
    env.close()
    pygame.quit()
