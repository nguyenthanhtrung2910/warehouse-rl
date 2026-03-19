from __future__ import annotations

import os
import time
import warnings
from typing import Any, Callable, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
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

# Common array size convention
# E - number of enviroments
# D - number of ended enviroments
# O - observation shape, can have multi dimensions
# A - number of agents
# AC - number actions
# B - batch size = E * A
# EP - number of episodes


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
        if self.memory is not None:
            with torch_train_mode(self.algorithm.policy):
                for _ in range(num_gradient_steps):
                    self.algorithm.update(buffer=self.memory, sample_size=batch_size)
        else:
            warnings.warn("Agent has no memory, nothing is updated.")
        return num_gradient_steps

    def get_act_batch(
        self,
        obs_e_a_o: npt.NDArray[np.float32],
        mask_e_a_ac: npt.NDArray[np.uint8],
        exploration_noise: bool,
    ) -> npt.NDArray[np.int_]:
        e = obs_e_a_o.shape[0]
        a = obs_e_a_o.shape[1]
        obs_b_o = obs_e_a_o.reshape(e * a, *obs_e_a_o.shape[2:])
        mask_b_ac = mask_e_a_ac.reshape(e * a, mask_e_a_ac.shape[2])
        obs_batch = cast(
            ObsBatchProtocol, Batch(obs=Batch(obs=obs_b_o, mask=mask_b_ac), info=None)
        )
        with torch.no_grad():
            act_b = self.algorithm.policy(obs_batch).act
        if exploration_noise:
            act_b = self.algorithm.policy.add_exploration_noise(act_b, obs_batch)
        return act_b.reshape(e, a)

    @staticmethod
    def get_act(
        policy: DiscreteQLearningPolicy[TNet],
        obs_a_o: npt.NDArray[np.float32],
        mask_a_ac: npt.NDArray[np.uint8],
        exploration_noise: bool,
    ):
        obs_batch = cast(
            ObsBatchProtocol, Batch(obs=Batch(obs=obs_a_o, mask=mask_a_ac), info=None)
        )
        with torch.no_grad():
            act_a = policy(obs_batch).act
        if exploration_noise:
            act_a = policy.add_exploration_noise(act_a, obs_batch)
        return act_a

    def save_to_memory(
        self,
        obs_e_a_o: npt.NDArray[np.float32],
        info_e: npt.NDArray[np.object_],
        obs_next_e_a_o: npt.NDArray[np.float32],
        act_e_a: npt.NDArray[np.int_],
        rew_e_a: npt.NDArray[np.float32],
        termination_e: npt.NDArray[np.bool_],
        truncation_e: npt.NDArray[np.bool_],
    ) -> None:
        if self.memory is not None:
            e = obs_e_a_o.shape[0]
            a = obs_e_a_o.shape[1]
            obs_b_o = obs_e_a_o.reshape(e * a, *obs_e_a_o.shape[2:])
            info_b = np.repeat(info_e, a)
            obs_next_b_o = obs_next_e_a_o.reshape(e * a, *obs_next_e_a_o.shape[2:])
            act_b = act_e_a.reshape(e * a)
            rew_b = rew_e_a.reshape(e * a)
            termination_b = np.repeat(termination_e, a)
            truncation_b = np.repeat(truncation_e, a)
            rollout = cast(
                RolloutBatchProtocol,
                Batch(
                    obs=obs_b_o,
                    info=info_b,
                    obs_next=obs_next_b_o,
                    act=act_b,
                    rew=rew_b,
                    terminated=termination_b,
                    truncated=truncation_b,
                ),
            )
            self.memory.add(rollout)
        else:
            warnings.warn("Agent has no memory, nothing is updated.")


class DecentralizedTrainer:
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
        n_agents: int,
        plot: bool = False,
    ) -> dict[str, Any]:
        assert agent.memory is not None, "Learning agent must having a memory."
        assert train_env.env_num * n_agents == agent.memory.buffer_num
        n_envs = train_env.env_num
        n_collected_steps = 0
        n_collected_episodes = 0
        n_gradient_steps = 0
        last_n_collected_steps = n_collected_steps
        last_n_collected_episodes = n_collected_episodes
        # Lists of recorded data for plotting
        episodes: list[int] = []
        rewards: list[float] = [0.0]
        start = time.time()
        obs_e, _ = train_env.reset()
        while n_collected_episodes < self.n_training_episodes:
            if self.train_fn:
                self.train_fn(n_collected_episodes, n_collected_steps)
            # Get observations from envs
            obs_e_a_o = np.stack([obs.obs for obs in obs_e])
            mask_e_a_ac = np.stack([obs.mask for obs in obs_e])
            # Forward observations to agent
            act_e_a = agent.get_act_batch(
                obs_e_a_o, mask_e_a_ac, exploration_noise=True
            )
            # Step in envs
            obs_next_e, rew_e_a, termination_e, truncation_e, info_e = train_env.step(
                act_e_a
            )
            obs_next_e_a_o = np.stack([obs.obs for obs in obs_next_e])
            # Add transitions to memories of all learning agents, only shared memory now
            agent.save_to_memory(
                obs_e_a_o,
                info_e,
                obs_next_e_a_o,
                act_e_a,
                rew_e_a,
                termination_e,
                truncation_e,
            )
            n_collected_steps += n_envs
            # Policy updating
            if (n_collected_steps - last_n_collected_steps) >= self.update_freq:
                num_bonus_steps = n_collected_steps - last_n_collected_steps
                with policy_within_training_step(agent.algorithm.policy):
                    n_gradient_steps += agent.policy_update_fn(
                        self.batch_size, num_bonus_steps
                    )
                last_n_collected_steps = n_collected_steps
            # Prepare new observation for next iteration
            obs_e = obs_next_e
            # Reset ended envs
            done_e = termination_e | truncation_e
            id_d = np.where(done_e == True)[0]  # noqa: E712
            if id_d.size != 0:
                obs_e[id_d], _ = train_env.reset(id_d)
                n_collected_episodes += id_d.size
            # Test
            if (n_collected_episodes - last_n_collected_episodes) >= self.test_freq:
                test_stats = self.test(test_env, n_agents, agent)
                num_steps, reward_metric = (
                    test_stats["mean_num_steps"],
                    test_stats["reward"],
                )
                if (
                    len(rewards) > 0
                    and reward_metric > rewards[-1]
                    and self.save_best_fn
                ):
                    self.save_best_fn(n_collected_episodes)
                episodes.append(n_collected_episodes)
                rewards.append(reward_metric)
                print(
                    "===episode {:04d} done with number steps: {:5.1f}, reward: {:+06.2f}===".format(
                        (n_collected_episodes), num_steps, reward_metric
                    )
                )
                last_n_collected_episodes = n_collected_episodes
                # Break if reach required reward
                if self.stop_fn and self.stop_fn(rewards[-1], n_collected_episodes):
                    break
        finish = time.time()
        if self.save_last_fn:
            self.save_last_fn()
        if plot:
            self.plot_stats(episodes, rewards)
        return {
            "reward_metric_stats": rewards,
            "num_collected_steps": n_collected_steps,
            "num_collected_episodes": n_collected_episodes,
            "num_gradient_steps": n_gradient_steps,
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
        n_agents: int,
        agent: OffPolicyAgent,
    ) -> dict[str, Any]:
        num_collected_steps = 0
        num_collected_episodes = 0
        num_envs = test_env.env_num
        # Array of size number episodes that stores reward in that episode
        rewards_p_a: list[npt.NDArray[np.float32]] = []
        rewards_e_a = np.zeros((num_envs, n_agents), dtype=np.float32)
        obs_e, _ = test_env.reset()
        while num_collected_episodes < self.n_testing_episodes:
            if self.test_fn:
                self.test_fn(num_collected_episodes, num_collected_steps)
            # Get observations from envs
            obs_e_a_o = np.stack([obs.obs for obs in obs_e])
            mask_e_a_ac = np.stack([obs.mask for obs in obs_e])
            # Forward observations to agent
            act_e_a = agent.get_act_batch(
                obs_e_a_o, mask_e_a_ac, exploration_noise=False
            )
            # Step in envs
            obs_next_e, rew_e_a, termination_e, truncation_e, _ = test_env.step(act_e_a)
            num_collected_steps += num_envs
            rewards_e_a += rew_e_a
            # Prepare new observation for next iteration
            obs_e = obs_next_e
            # Reset ended envs
            done_e = termination_e | truncation_e
            id_d = np.where(done_e == True)[0]  # noqa: E712
            if id_d.size != 0:
                obs_e[id_d], _ = test_env.reset(id_d)
                num_collected_episodes += id_d.size
                # Save reward from ended envs and start new reward recording
                rewards_p_a.append(rewards_e_a[id_d])
                rewards_e_a[id_d] = 0
        reward_p_a = np.vstack(rewards_p_a)
        if self.reward_metric:
            reward = self.reward_metric(reward_p_a)
        else:
            reward = reward_p_a.mean()
        return {
            "reward": reward,
            "mean_num_steps": num_collected_steps / num_collected_episodes,
            "num_collected_steps": num_collected_steps,
            "num_collected_episodes": num_collected_episodes,
        }
