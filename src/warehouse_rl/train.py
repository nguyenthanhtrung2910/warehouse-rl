import math
import os
from typing import Any, Callable, cast

import numpy as np
import torch
from gymnasium import spaces
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import (
    Batch,
    CollectStats,
    PrioritizedVectorReplayBuffer,
    to_numpy,  # pyright: ignore[reportUnknownVariableType]
)
from tianshou.data.collector import Collector
from tianshou.data.types import ModelOutputBatchProtocol, ObsBatchProtocol
from tianshou.env.venvs import DummyVectorEnv
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from torch.nn.modules import Module

from warehouse_rl.warehouse import Warehouse


class DQNPolicyWithActionMask(DiscreteQLearningPolicy[torch.nn.Module]):
    def forward(
        self,
        batch: ObsBatchProtocol,
        state: Any | None = None,
        model: Module | None = None,
    ) -> ModelOutputBatchProtocol:
        if model is None:
            model = self.model
        obs = batch.obs
        info = batch.info
        if info[0] is not None:
            mask = np.array([dictionary["mask"] for dictionary in info])  # pyright: ignore[reportArgumentType]
        else:
            mask = None
        action_values_BA, hidden_BH = model(obs, state=state, info=info)
        q = self.compute_q_value(action_values_BA, mask)
        act_B = to_numpy(q.argmax(dim=1))
        result = Batch(logits=action_values_BA, act=act_B, state=hidden_BH)
        return cast(ModelOutputBatchProtocol, result)


def exponential_annealing(
    begin: float, end: float, decay_factor: float
) -> Callable[[float], float]:
    return lambda episode: max(begin * decay_factor**episode, end)


def natural_exponential_annealing(
    begin: float, end: float, rate: float
) -> Callable[[float], float]:
    return lambda episode: end + (begin - end) * math.exp(-rate * episode)


if __name__ == "__main__":
    eps_schedule = exponential_annealing(1.0, 0.05, 0.996)
    beta_schedule = natural_exponential_annealing(0.4, 1.0, 0.007)

    envs = DummyVectorEnv([lambda: Warehouse(2, 2, 2, 2, True, 500) for _ in range(16)])
    test_envs = DummyVectorEnv(
        [lambda: Warehouse(2, 2, 2, 2, True, 500) for _ in range(16)]
    )
    net = Net(
        state_shape=3,
        action_shape=4,
        hidden_sizes=[512, 512, 256, 256, 128, 64],
        norm_layer=torch.nn.LayerNorm,
        dueling_param=(
            {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
            {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
        ),
    )
    policy = DQNPolicyWithActionMask(
        model=net, action_space=spaces.Discrete(4), eps_training=1.0
    )
    algorithm = DQN[DQNPolicyWithActionMask](
        policy=policy,
        optim=AdamOptimizerFactory(lr=0.0001),
        gamma=0.99,
        n_step_return_horizon=15,
        target_update_freq=300,
        is_double=True,
    )

    buffer = PrioritizedVectorReplayBuffer(
        total_size=1_000_000,
        buffer_num=16,
        alpha=0.6,
        beta=0.4,
    )
    training_collector = Collector[CollectStats](
        policy=algorithm, env=envs, buffer=buffer, exploration_noise=True
    )
    test_collector = Collector[CollectStats](
        policy=algorithm,
        env=test_envs,
        exploration_noise=False,
    )

    FOLDER = "DQNagent"
    os.makedirs(os.path.join(os.getcwd(), FOLDER), exist_ok=True)

    def train_fn(num_epoch: int, step_idx: int) -> None:
        if step_idx % 400 == 0:
            policy.set_eps_training(eps_schedule(step_idx / 400))
            buffer.set_beta(beta_schedule(step_idx / 400))

    def stop_fn(mean_rewards: float) -> bool:
        return False

    result = algorithm.run_training(
        OffPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=3,
            epoch_num_steps=100_000,
            collection_step_num_env_steps=96,
            test_step_num_episodes=16,
            batch_size=64,
            update_step_num_gradient_steps_per_sample=0.01,
            test_in_training=True,
            training_fn=train_fn,
            stop_fn=stop_fn,
        )
    )

    torch.save(policy.state_dict(), os.path.join(FOLDER, "last.pth"))
    print(result)
