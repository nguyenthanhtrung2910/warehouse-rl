from __future__ import annotations

import math
import os
import typing

import gymnasium
import tianshou.algorithm.modelfree.dqn
import tianshou.algorithm.optim
import tianshou.data.buffer.vecbuf
import tianshou.env
import tianshou.utils.net.common
import torch

import warehouse_rl.agents
import warehouse_rl.warehouse_b

n_agents = 3
net = tianshou.utils.net.common.Net(
    state_shape=36 + 3,
    action_shape=4,
    hidden_sizes=[1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64],
    norm_layer=torch.nn.LayerNorm,
    activation=torch.nn.ReLU,
    dueling_param=(
        {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
        {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
    ),
)
policy: tianshou.algorithm.modelfree.dqn.DiscreteQLearningPolicy[
    tianshou.utils.net.common.Net
] = tianshou.algorithm.modelfree.dqn.DiscreteQLearningPolicy(
    model=net, action_space=gymnasium.spaces.Discrete(4), eps_training=1.0
)
algorithm: tianshou.algorithm.modelfree.dqn.DQN[
    tianshou.algorithm.modelfree.dqn.DiscreteQLearningPolicy[
        tianshou.utils.net.common.Net
    ]
] = tianshou.algorithm.modelfree.dqn.DQN(
    policy=policy,
    optim=tianshou.algorithm.optim.AdamOptimizerFactory(lr=0.0001),
    gamma=0.99,
    n_step_return_horizon=20,
    target_update_freq=400,
    is_double=True,
)
memory = tianshou.data.buffer.vecbuf.PrioritizedVectorReplayBuffer(
    total_size=500_000 * n_agents,
    buffer_num=16 * n_agents,
    alpha=0.6,
    beta=0.4,
)
agent = warehouse_rl.agents.OffPolicyAgent(
    algorithm,
    memory=memory,
    gradient_steps_per_env_step=0.02,
)


def exponential_annealing(
    begin: float, end: float, decay_factor: float
) -> typing.Callable[[int], float]:
    return lambda episode: max(begin * decay_factor**episode, end)


def natural_exponential_annealing(
    begin: float, end: float, rate: float
) -> typing.Callable[[int], float]:
    return lambda episode: end + (begin - end) * math.exp(-rate * episode)


eps_schedule: typing.Callable[[int], float] = exponential_annealing(1.0, 0.05, 0.997)
beta_schedule: typing.Callable[[int], float] = natural_exponential_annealing(
    0.4, 1.0, 0.006
)
ckpt_dir = "ckpt/b"
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


train_env = tianshou.env.DummyVectorEnv(
    [
        lambda: warehouse_rl.warehouse_b.WarehouseB(
            2, 2, 3, 3, True, 750, n_agents, 20, 6
        )
        for _ in range(16)
    ]
)
test_env = tianshou.env.DummyVectorEnv(
    [
        lambda: warehouse_rl.warehouse_b.WarehouseB(
            2, 2, 3, 3, True, 750, n_agents, 20, 6
        )
        for _ in range(16)
    ]
)

trainer = warehouse_rl.agents.DecentralizedTrainer(
    batch_size=64,
    update_freq=200,
    test_freq=50,
    n_training_episodes=1000,
    n_testing_episodes=32,
    train_fn=train_fn,
    save_last_fn=save_last_fn,
    save_best_fn=save_best_fn,
)

trainer.train(train_env, test_env, agent, n_agents, True)
