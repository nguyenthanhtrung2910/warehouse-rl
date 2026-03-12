from typing import Any, cast
from gymnasium import spaces
import numpy as np

from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.data.types import ModelOutputBatchProtocol, ObsBatchProtocol
from tianshou.data import Batch, to_numpy  # pyright: ignore[reportUnknownVariableType]
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.algorithm.modelfree.dqn import DQN
from tianshou.data.collector import Collector
from tianshou.data import CollectStats
from tianshou.data.buffer.vecbuf import VectorReplayBuffer
from tianshou.utils.net.common import Net
import torch
from torch.nn.modules import Module

from tianshou.env.venvs import DummyVectorEnv
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
        mask = np.array([dic["mask"] for dic in info])  # pyright: ignore[reportArgumentType]
        action_values_BA, hidden_BH = model(obs, state=state, info=info)
        q = self.compute_q_value(action_values_BA, mask)
        act_B = to_numpy(q.argmax(dim=1))
        result = Batch(logits=action_values_BA, act=act_B, state=hidden_BH)
        return cast(ModelOutputBatchProtocol, result)


envs = DummyVectorEnv([lambda: Warehouse(2, 2, 2, 2, True, 500) for _ in range(8)])

net = Net(state_shape=3, action_shape=4, hidden_sizes=[128, 128, 128])
policy = DQNPolicyWithActionMask(
    model=net, action_space=spaces.Discrete(4), eps_training=1.0
)

algorithm = DQN[DQNPolicyWithActionMask](
    policy=policy,
    optim=AdamOptimizerFactory(lr=0.0001),
    gamma=0.9,
    n_step_return_horizon=3,
    target_update_freq=300,
)
buffers = VectorReplayBuffer(32, 8)
training_collector = Collector[CollectStats](
    algorithm,
    envs,
    buffers,
    exploration_noise=True,
)

collect_result = training_collector.collect(reset_before_collect=True, n_step=32)
print(collect_result)
print(buffers)
