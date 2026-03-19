from __future__ import annotations

import math
import os
from typing import Callable

import torch
from gymnasium import spaces
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data.buffer.vecbuf import PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net

from warehouse_rl.agents import DecentralizedTrainer, OffPolicyAgent
from warehouse_rl.warehouse_b import WarehouseB


n_agents = 4
net = Net(
    state_shape=19,
    action_shape=4,
    hidden_sizes=[1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64],
    norm_layer=torch.nn.LayerNorm,
    activation=torch.nn.ReLU,
    dueling_param=(
        {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
        {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
    ),
)
policy = DiscreteQLearningPolicy(
    model=net, action_space=spaces.Discrete(4), eps_training=1.0
)
algorithm = DQN(
    policy=policy,
    optim=AdamOptimizerFactory(lr=0.0001),
    gamma=0.99,
    n_step_return_horizon=20,
    target_update_freq=400,
    is_double=True,
)
memory = PrioritizedVectorReplayBuffer(
    total_size=250_000 * n_agents,
    buffer_num=16 * n_agents,
    alpha=0.6,
    beta=0.4,
)
agent = OffPolicyAgent(
    algorithm,
    memory=memory,
    gradient_steps_per_env_step=0.02,
)


def exponential_annealing(
    begin: float, end: float, decay_factor: float
) -> Callable[[int], float]:
    return lambda episode: max(begin * decay_factor**episode, end)


def natural_exponential_annealing(
    begin: float, end: float, rate: float
) -> Callable[[int], float]:
    return lambda episode: end + (begin - end) * math.exp(-rate * episode)


eps_schedule = exponential_annealing(1.0, 0.05, 0.995)
beta_schedule = natural_exponential_annealing(0.4, 1.0, 0.01)
ckpt_dir = "ckpt"
os.makedirs(os.path.join(os.getcwd(), ckpt_dir), exist_ok=True)


def train_fn(episode: int, step: int) -> None:
    agent.algorithm.policy.set_eps_training(eps_schedule(episode))
    if agent.memory is not None:
        agent.memory.set_beta(beta_schedule(episode))


def save_last_fn() -> None:
    torch.save(agent.algorithm.policy.state_dict(), os.path.join(ckpt_dir, "last.pth"))
    torch.save(agent.algorithm.optim.state_dict(), os.path.join(ckpt_dir, "optim.pth"))  # type: ignore


def save_best_fn(episode: int) -> None:
    torch.save(agent.algorithm.policy.state_dict(), os.path.join(ckpt_dir, "best.pth"))


train_env = DummyVectorEnv(
    [lambda: WarehouseB(2, 2, 2, 2, True, 500, n_agents, 12, 4) for _ in range(16)]
)
test_env = DummyVectorEnv(
    [lambda: WarehouseB(2, 2, 2, 2, True, 500, n_agents, 12, 4) for _ in range(16)]
)

trainer = DecentralizedTrainer(
    batch_size=64,
    update_freq=200,
    test_freq=50,
    n_training_episodes=500,
    n_testing_episodes=32,
    train_fn=train_fn,
    save_last_fn=save_last_fn,
    save_best_fn=save_best_fn,
)

trainer.train(train_env, test_env, agent, n_agents, True)
