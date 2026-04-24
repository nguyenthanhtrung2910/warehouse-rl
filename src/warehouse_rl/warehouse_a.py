from __future__ import annotations

import dataclasses
import random
import typing

import cv2
import gymnasium
import gymnasium.core
import numpy as np
import pygame
import pygame.math

import warehouse_rl.enums
import warehouse_rl.map
import warehouse_rl.sprites
import warehouse_rl.warehouse

SPEED = 250


@dataclasses.dataclass
class Observation:
    obs_a_o: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    mask_a_ac: np.ndarray[tuple[int, int], np.dtype[np.unsignedinteger]]


class Warehouse(
    gymnasium.core.Env[Observation, np.ndarray[tuple[int], np.dtype[np.integer]]]
):
    step_counter: int
    parcel_counter: int
    norequested_step_counter: int
    n_steps: int
    n_shuttles: int
    map: warehouse_rl.map.WarehouseMap
    shuttles: list[warehouse_rl.sprites.Combined]
    in_line_parcels: list[warehouse_rl.sprites.Parcel]
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
        request_freq: int = 18,
        render_mode: warehouse_rl.enums.RenderMode = warehouse_rl.enums.RenderMode.Null,
        observation_mode: warehouse_rl.enums.ObsMode = warehouse_rl.enums.ObsMode.Flatten,
        recording: bool = False,
    ) -> None:
        super().__init__()
        self.step_counter = 0
        self.parcel_counter = 0
        self.norequested_step_counter = 0
        self.n_steps = n_steps  # const
        self.n_shuttles = n_shuttles  # const
        n_rays: int = 2 if is_double_line else 1
        self.map = warehouse_rl.map.WarehouseMap(
            n_rows, n_columns, n_subrows, n_lines, n_rays
        )
        self.size = n_rows * n_columns * n_subrows * n_lines
        self.in_line_parcels = []
        self.shuttles = []
        for ray_node in random.sample(list(self.map.ray_nodes.values()), n_shuttles):
            self.shuttles.append(
                warehouse_rl.sprites.Combined(ray_node, self.map.map_size)
            )
        self.request_freq = request_freq  # const
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
                "warehouse.mp4", fourcc, 15, self.map.image.get_size()
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
        self.norequested_step_counter = 0
        # Reset all shuttle
        for shuttle, ray_node in zip(
            self.shuttles,
            random.sample(list(self.map.ray_nodes.values()), self.n_shuttles),
        ):
            shuttle.reset(ray_node)
        for line_node in self.map.line_nodes.values():
            line_node.parcel = None
        _ = warehouse_rl.sprites.Parcel(self.map.line_nodes[f"1.{-1}"])
        self.in_line_parcels.clear()
        self.render()
        obs: Observation = self.__make_observation()
        info: dict[str, typing.Any] = {}
        return obs, info

    @typing.override
    def step(
        self, action: np.ndarray[tuple[int], np.dtype[np.integer]]
    ) -> tuple[
        Observation,
        np.ndarray[tuple[int], np.dtype[np.floating]],
        bool,
        bool,
        dict[str, typing.Any],
    ]:
        if self.step_counter == self.n_steps:
            raise ValueError(
                "The environment has ended. You have to reset it before step it further."
            )
        reward_a: list[float] = [0.0] * (self.n_shuttles)
        indices: list[int] = list(range(len(self.shuttles)))
        # Get a random order of shuttles
        random.shuffle(indices)
        # Perform shuttles moves
        shuttle_movements: list[warehouse_rl.warehouse.Movement] = []
        for i in indices:
            result: warehouse_rl.sprites.StepResult = self.shuttles[i].step(
                warehouse_rl.enums.Action(action[i])
            )
            reward_a[i] += result.reward
            if result.movements:
                shuttle_movements.extend(result.movements)
        self.__simulate_movement(shuttle_movements)
        # TODO: If we want parcel movement is parallel with shuttle movement,
        # we have to add new action. Pick or drop parcel right away after shuttle movement
        # and simulate all is not right because the target which sprite move to can't move
        # during its movement.
        parcel_movements: list[warehouse_rl.warehouse.Movement] = []
        for i in indices:
            warehouse_is_full = (
                len(self.in_line_parcels) > self.map.n_line_nodes - self.n_shuttles
            )
            result: warehouse_rl.sprites.StepResult = self.shuttles[i].pick_up(
                warehouse_is_full
            )
            reward_a[i] += result.reward
            if result.parcel and result.parcel in self.in_line_parcels:
                self.in_line_parcels.remove(result.parcel)
            if result.movements:
                parcel_movements.extend(result.movements)
            result: warehouse_rl.sprites.StepResult = self.shuttles[i].drop_off()
            reward_a[i] += result.reward
            if result.parcel and not result.parcel.is_requested:
                self.in_line_parcels.append(result.parcel)
            if result.movements:
                parcel_movements.extend(result.movements)
        self.__simulate_movement(parcel_movements)
        # Delete deliveried parcel
        if self.map.palletized_node.parcel:
            self.parcel_counter += 1
            self.map.palletized_node.parcel = None
        self.__request_parcel()
        self.step_counter += 1
        obs: Observation = self.__make_observation()
        termination: bool = self.step_counter == self.n_steps
        truncation: bool = False
        info: dict[str, typing.Any] = {}
        return (
            obs,
            np.array(reward_a),
            termination,
            truncation,
            info,
        )

    def __request_parcel(self):
        requested_parcels = [
            parcel for parcel in self.in_line_parcels if parcel.is_requested
        ]
        if (
            self.norequested_step_counter >= self.request_freq
            and len(requested_parcels) < 10
        ):
            unrequested_parcels = [
                parcel for parcel in self.in_line_parcels if not parcel.is_requested
            ]
            if unrequested_parcels:
                parcel = unrequested_parcels[random.randrange(len(unrequested_parcels))]
                parcel.is_requested = True
            self.norequested_step_counter = 0
            return
        self.norequested_step_counter += 1

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
                # obs <==> obs_a_o
                line_nodes_states: list[float] = []
                for line_node in self.map.line_nodes.values():
                    if not line_node.is_depalletized and not line_node.is_palletized:
                        if line_node.parcel:
                            line_nodes_states.append(
                                1.0 if line_node.parcel.is_requested else 0.5
                            )
                        else:
                            line_nodes_states.append(0.0)
                obs: np.ndarray[tuple[int, int], np.dtype[np.floating]] = np.vstack(
                    [
                        np.hstack((shuttle.state, np.array(line_nodes_states)))
                        for shuttle in self.shuttles
                    ]
                )
            case warehouse_rl.enums.ObsMode.ResizedWindow:
                # TODO: obs <==> obs_a_c_h_w
                obs = self.__create_obs_img()
            case _:
                raise ValueError(
                    f"Invalid render_mode value: {self.__observation_mode}."
                )
        mask_a_ac: np.ndarray[tuple[int, int], np.dtype[np.unsignedinteger]] = (
            np.vstack([shuttle.mask for shuttle in self.shuttles])
        )
        return Observation(obs, mask_a_ac)

    def __simulate_movement(
        self, movements: list[warehouse_rl.warehouse.Movement]
    ) -> None:
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
        500,
        3,
        18,
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
        env.render()
        mask_a_ac = obs.mask_a_ac
        action_a: list[int] = []
        for mask_ac in mask_a_ac:
            legal_action_ac: list[int] = [i for i, v in enumerate(mask_ac) if v]
            if len(legal_action_ac) != 0:
                action_a.append(random.choice(legal_action_ac))
            else:
                action_a.append(0)
        next_obs, reward_a, termination, truncation, info = env.step(np.array(action_a))
        print(
            f"In step {env.n_steps}: observation {obs.obs_a_o}, action {action_a} reward {reward_a}"
        )
        obs: Observation = next_obs
        done: bool = termination or truncation
    env.close()
    pygame.quit()
