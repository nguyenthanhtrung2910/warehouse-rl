from __future__ import annotations

import os
import time
import warnings
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from typing import TypeVar
import torch
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.data import Batch
from tianshou.data.buffer.vecbuf import PrioritizedVectorReplayBuffer
from tianshou.data.types import ObsBatchProtocol, RolloutBatchProtocol
from tianshou.env import DummyVectorEnv
from tianshou.utils.torch_utils import (
    policy_within_training_step,
    torch_train_mode,
)

TNet = TypeVar("TNet", bound=torch.nn.Module)

class OffPolicyAgent:
    def __init__(
        self,
        algorithm: DQN[DiscreteQLearningPolicy[TNet]],
        memory: PrioritizedVectorReplayBuffer | None = None,
        gradient_steps_per_env_step: float = 1.0,
    ) -> None:
        self.algorithm = algorithm
        self.memory = memory
        self.gradient_steps_per_env_step = gradient_steps_per_env_step

        # Policy should be always in eval mode to inference action
        # Training mode is turned on only within context manager
        self.algorithm.policy.eval()

    def policy_update_fn(self, batch_size: int, num_collected_steps: int):
        num_gradient_steps = round(
            self.gradient_steps_per_env_step * num_collected_steps
        )
        if num_gradient_steps == 0:
            raise ValueError(
                f"n_gradient_steps is 0, n_collected_steps={num_collected_steps}, "
                f"update_per_step={self.gradient_steps_per_env_step}",
            )
        if self.memory:
            with torch_train_mode(self.algorithm.policy):
                for _ in range(num_gradient_steps):
                    self.algorithm.update(buffer=self.memory, sample_size=batch_size)
        else:
            warnings.warn("Agent has no memory, nothing is updated.")
        return num_gradient_steps

    def get_act_batch(self, obs_batch: ObsBatchProtocol, exploration_noise: bool):
        with torch.no_grad():
            act = self.algorithm.policy(obs_batch).act
        if exploration_noise:
            act = self.algorithm.policy.add_exploration_noise(act, obs_batch)
        return act


class Trainer:
    def __init__(
        self,
        batch_size: int = 64,
        update_freq: int = 100,
        test_freq: int = 100,
        n_training_episodes: int = 5000,
        n_testing_episodes: int = 50,
        train_fn: Callable[[int, int], None] | None = None,
        test_fn: Callable[[int, int], None] | None = None,
        save_best_fn: Callable[[int], None] | None = None,
        save_last_fn: Callable[[], None] | None = None,
        stop_fn: Callable[[float, int], bool] | None = None,
        reward_metric: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.test_freq = test_freq
        self.n_training_episodes = n_training_episodes
        self.n_testing_episodes = n_testing_episodes
        self.train_fn = train_fn
        self.test_fn = test_fn
        self.save_best_fn = save_best_fn
        self.save_last_fn = save_last_fn
        self.stop_fn = stop_fn
        self.reward_metric = reward_metric

    def train(
        self,
        train_env: DummyVectorEnv,
        test_env: DummyVectorEnv,
        agent: OffPolicyAgent,
        plot: bool = False,
    ) -> dict[str, Any]:
        # E - number of enviroments
        # R - number of running envs
        # O - observation-vector size
        # A - number actions
        assert agent.memory is not None, "Learning agent must having a memory."
        assert train_env.env_num == agent.memory.buffer_num
        num_envs = train_env.env_num
        num_collected_steps = 0
        num_collected_episodes = 0
        num_gradient_steps = 0
        last_num_collected_steps = num_collected_steps
        last_num_collected_episodes = num_collected_episodes

        # lists to record data for plotting
        episodes: list[int] = []
        rewards: list[float] = [0.0]
        start = time.time()
        while num_collected_episodes < self.n_training_episodes:
            done_e = np.zeros(num_envs, dtype=np.bool_)
            obs_e, _ = train_env.reset()
            if self.train_fn:
                self.train_fn(num_collected_episodes, num_collected_steps)
            while not all(done_e):
                # Get observations from running envs
                ids_r = np.where(done_e == False)[0]  # noqa: E712
                obs_r = obs_e[ids_r]
                obs_r_o = np.array([obs.obs for obs in obs_r])
                action_mask_r_a = np.array([obs.mask for obs in obs_r])
                obs_r = Batch(obs=Batch(obs=obs_r_o, mask=action_mask_r_a), info=None)
                obs_r = cast(ObsBatchProtocol, obs_r)

                # Forward observations to agent
                act_r = agent.get_act_batch(obs_r, exploration_noise=True)

                # Step in running envs
                next_obs_r, rew_r, terminated_r, truncated_r, info_r = train_env.step(
                    act_r, ids_r
                )
                next_obs_r_o = np.array([obs.obs for obs in next_obs_r])

                # Add transitions to memories of all learning agents, only shared memory now
                rollout = cast(
                    RolloutBatchProtocol,
                    Batch(
                        obs=obs_r_o,
                        act=act_r,
                        rew=rew_r,
                        terminated=terminated_r,
                        truncated=truncated_r,
                        obs_next=next_obs_r_o,
                        info=info_r,
                    ),
                )
                agent.memory.add(rollout, buffer_ids=ids_r)  # type: ignore
                num_collected_steps += ids_r.size

                # Policy updating
                if (num_collected_steps - last_num_collected_steps) >= self.update_freq:
                    num_bonus_steps = num_collected_steps - last_num_collected_steps
                    with policy_within_training_step(agent.algorithm.policy):
                        num_gradient_steps += agent.policy_update_fn(
                            self.batch_size, num_bonus_steps
                        )
                    last_num_collected_steps = num_collected_steps

                # Observe new observations and dones of all envs
                done_e[ids_r] = terminated_r | truncated_r
                obs_e[ids_r] = next_obs_r

            num_collected_episodes += num_envs

            # Test
            if (num_collected_episodes - last_num_collected_episodes) >= self.test_freq:
                test_stats = self.test(test_env, agent)
                num_steps, reward_metric = (
                    test_stats["mean_num_steps"],
                    test_stats["reward"],
                )
                if (
                    len(rewards) > 0
                    and reward_metric > rewards[-1]
                    and self.save_best_fn
                ):
                    self.save_best_fn(num_collected_episodes)
                episodes.append(num_collected_episodes)
                rewards.append(reward_metric)
                print(
                    "===episode {:04d} done with number steps: {:5.1f}, reward: {:+06.2f}===".format(
                        (num_collected_episodes), num_steps, reward_metric
                    )
                )
                last_num_collected_episodes = num_collected_episodes

                # Break if reach required reward
                if self.stop_fn and self.stop_fn(rewards[-1], num_collected_episodes):
                    break

        finish = time.time()
        if self.save_last_fn:
            self.save_last_fn()

        if plot:
            self.plot_stats(episodes, rewards)

        return {
            "reward_metric_stats": rewards,
            "num_collected_steps": num_collected_steps,
            "num_collected_episodes": num_collected_episodes,
            "num_gradient_steps": num_gradient_steps,
            "training_time": finish - start,
        }

    @staticmethod
    def plot_stats(episodes: list[int], rewards: list[float]) -> None:
        rewards.pop(0)
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))  # type: ignore
        axes.plot(  # type: ignore
            np.array(episodes), np.array(rewards), marker=".", color="b", label="reward"
        )
        axes.set_xlabel("Number of collected episodes")  # type: ignore
        axes.set_ylabel("Reward metric")  # type: ignore
        axes.set_title("Performance of agent through episode.")  # type: ignore
        axes.legend()  # type: ignore
        axes.grid()  # type: ignore
        fig.tight_layout()
        os.makedirs(os.path.join(os.getcwd(), "plots"), exist_ok=True)
        fig.savefig(  # type: ignore
            os.path.join(os.getcwd(), "plots", "results.png"),
            dpi=150,
            bbox_inches="tight",
        )

    def test(
        self,
        test_env: DummyVectorEnv,
        agent: OffPolicyAgent,
    ) -> dict[str, Any]:
        # P - number of episodes
        # E - number of enviroments
        # R - number of running envs
        # O - observation-vector size
        # A - number actions
        num_collected_steps = 0
        num_collected_episodes = 0
        num_envs = test_env.env_num
        rewards_p_e = np.empty((0, num_envs))

        while num_collected_episodes < self.n_testing_episodes:
            rewards_e = np.zeros(num_envs)
            done_e = np.zeros(num_envs, dtype=np.bool_)
            obs_e, _ = test_env.reset()
            if self.test_fn:
                self.test_fn(num_collected_episodes, num_collected_steps)
            while not all(done_e):
                # Gets observations of running envs
                ids_r = np.where(done_e == False)[0]  # noqa: E712
                obs_r = obs_e[ids_r]
                obs_r_o = np.array([obs.obs for obs in obs_r])
                action_mask_r_a = np.array([obs.mask for obs in obs_r])
                obs_r = Batch(obs=Batch(obs=obs_r_o, mask=action_mask_r_a), info=None)
                obs_r = cast(ObsBatchProtocol, obs_r)

                # Forward observations to agent
                act_r = agent.get_act_batch(obs_r, exploration_noise=False)

                # Step in the running envs
                next_obs_r, rew_r, termination_r, truncation_r, _ = test_env.step(
                    act_r, ids_r
                )
                rewards_e[ids_r] += rew_r
                num_collected_steps += ids_r.size

                # Observe new observations and dones of all envs
                done_e[ids_r] = termination_r | truncation_r
                obs_e[ids_r] = next_obs_r

            num_collected_episodes += num_envs
            rewards_p_e = np.vstack((rewards_p_e, rewards_e))

        if self.reward_metric:
            reward = self.reward_metric(rewards_p_e)
        else:
            reward = rewards_p_e.mean()
        return {
            "reward": reward,
            "mean_num_steps": num_collected_steps / num_collected_episodes,
            "num_collected_steps": num_collected_steps,
            "num_collected_episodes": num_collected_episodes,
        }
