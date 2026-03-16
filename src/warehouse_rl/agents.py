from __future__ import annotations

import os
import time
import warnings
from typing import Any, Callable, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
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
        # D - number of done envs
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
        # Lists to record data for plotting
        episodes: list[int] = []
        rewards: list[float] = [0.0]
        start = time.time()
        obs_e, _ = train_env.reset()
        while num_collected_episodes < self.n_training_episodes:
            if self.train_fn:
                self.train_fn(num_collected_episodes, num_collected_steps)
            # Get observations from envs
            obs_e_o = np.array([obs.obs for obs in obs_e])
            action_mask_e_a = np.array([obs.mask for obs in obs_e])
            obs_e = Batch(obs=Batch(obs=obs_e_o, mask=action_mask_e_a), info=None)
            obs_e = cast(ObsBatchProtocol, obs_e)
            # Forward observations to agent
            act_e = agent.get_act_batch(obs_e, exploration_noise=True)
            # Step in envs
            next_obs_e, rew_e, termination_e, truncation_e, info_e = train_env.step(
                act_e
            )
            next_obs_e_o = np.array([obs.obs for obs in next_obs_e])
            # Add transitions to memories of all learning agents, only shared memory now
            rollout = cast(
                RolloutBatchProtocol,
                Batch(
                    obs=obs_e_o,
                    act=act_e,
                    rew=rew_e,
                    terminated=termination_e,
                    truncated=truncation_e,
                    obs_next=next_obs_e_o,
                    info=info_e,
                ),
            )
            agent.memory.add(rollout)
            num_collected_steps += num_envs
            # Policy updating
            if (num_collected_steps - last_num_collected_steps) >= self.update_freq:
                num_bonus_steps = num_collected_steps - last_num_collected_steps
                with policy_within_training_step(agent.algorithm.policy):
                    num_gradient_steps += agent.policy_update_fn(
                        self.batch_size, num_bonus_steps
                    )
                last_num_collected_steps = num_collected_steps
            # Prepare new observation for next iteration
            obs_e = next_obs_e
            # Reset ended envs
            done_e = termination_e | truncation_e
            id_d = np.where(done_e == True)[0]  # noqa: E712
            if id_d.size != 0:
                obs_e[id_d], _ = train_env.reset(id_d)
                num_collected_episodes += id_d.size
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
        # O - observation-vector size
        # A - number actions
        num_collected_steps = 0
        num_collected_episodes = 0
        num_envs = test_env.env_num
        # Array of size number episodes that stores reward in that episode
        rewards_p = np.array([])
        rewards_e = np.zeros(num_envs)
        obs_e, _ = test_env.reset()
        while num_collected_episodes < self.n_testing_episodes:
            if self.test_fn:
                self.test_fn(num_collected_episodes, num_collected_steps)
            # Gets observations of running envs
            obs_e_o = np.array([obs.obs for obs in obs_e])
            action_mask_e_a = np.array([obs.mask for obs in obs_e])
            obs_e = Batch(obs=Batch(obs=obs_e_o, mask=action_mask_e_a), info=None)
            obs_e = cast(ObsBatchProtocol, obs_e)
            # Forward observations to agent
            act_e = agent.get_act_batch(obs_e, exploration_noise=True)
            # Step in running envs
            next_obs_e, rew_e, termination_e, truncation_e, _ = test_env.step(act_e)
            num_collected_steps += num_envs
            rewards_e += rew_e
            # Prepare new observation for next iteration
            obs_e = next_obs_e
            # Reset ended envs
            done_e = termination_e | truncation_e
            id_d = np.where(done_e == True)[0]  # noqa: E712
            if id_d.size != 0:
                obs_e[id_d], _ = test_env.reset(id_d)
                num_collected_episodes += id_d.size
                # Save reward from ended envs and start new reward recording
                rewards_p = np.hstack((rewards_p, rewards_e[id_d]))
                rewards_e[id_d] = 0
        if self.reward_metric:
            reward = self.reward_metric(rewards_p)
        else:
            reward = rewards_p.mean()
        return {
            "reward": reward,
            "mean_num_steps": num_collected_steps / num_collected_episodes,
            "num_collected_steps": num_collected_steps,
            "num_collected_episodes": num_collected_episodes,
        }
