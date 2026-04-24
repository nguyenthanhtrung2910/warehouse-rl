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
import warehouse_rl.warehouse

n_loaders = 2
n_pickers = 2
net_loader = tianshou.utils.net.common.Net(
    state_shape=36 + 7,
    action_shape=4,
    hidden_sizes=[1024, 1024, 512, 512, 256, 256, 128, 128, 128, 64, 64, 64],
    norm_layer=torch.nn.LayerNorm,
    activation=torch.nn.ReLU,
    dueling_param=(
        {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
        {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
    ),
)
net_picker = tianshou.utils.net.common.Net(
    state_shape=36 + 7,
    action_shape=4,
    hidden_sizes=[1024, 1024, 512, 512, 256, 256, 128, 128, 128, 64, 64, 64],
    norm_layer=torch.nn.LayerNorm,
    activation=torch.nn.ReLU,
    dueling_param=(
        {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
        {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
    ),
)
loader_policy: tianshou.algorithm.modelfree.dqn.DiscreteQLearningPolicy[
    tianshou.utils.net.common.Net
] = tianshou.algorithm.modelfree.dqn.DiscreteQLearningPolicy(
    model=net_loader, action_space=gymnasium.spaces.Discrete(4), eps_training=1.0
)
picker_policy: tianshou.algorithm.modelfree.dqn.DiscreteQLearningPolicy[
    tianshou.utils.net.common.Net
] = tianshou.algorithm.modelfree.dqn.DiscreteQLearningPolicy(
    model=net_picker, action_space=gymnasium.spaces.Discrete(4), eps_training=1.0
)
loader_algorithm: tianshou.algorithm.modelfree.dqn.DQN[
    tianshou.algorithm.modelfree.dqn.DiscreteQLearningPolicy[
        tianshou.utils.net.common.Net
    ]
] = tianshou.algorithm.modelfree.dqn.DQN(
    policy=loader_policy,
    optim=tianshou.algorithm.optim.AdamOptimizerFactory(lr=0.0001),
    gamma=0.99,
    n_step_return_horizon=30,
    target_update_freq=500,
    is_double=True,
)
picker_algorithm: tianshou.algorithm.modelfree.dqn.DQN[
    tianshou.algorithm.modelfree.dqn.DiscreteQLearningPolicy[
        tianshou.utils.net.common.Net
    ]
] = tianshou.algorithm.modelfree.dqn.DQN(
    policy=picker_policy,
    optim=tianshou.algorithm.optim.AdamOptimizerFactory(lr=0.0001),
    gamma=0.99,
    n_step_return_horizon=30,
    target_update_freq=500,
    is_double=True,
)
loader_memory = tianshou.data.buffer.vecbuf.PrioritizedVectorReplayBuffer(
    total_size=300_000 * n_loaders,
    buffer_num=16 * n_loaders,
    alpha=0.6,
    beta=0.4,
)
picker_memory = tianshou.data.buffer.vecbuf.PrioritizedVectorReplayBuffer(
    total_size=600_000 * n_pickers,
    buffer_num=16 * n_pickers,
    alpha=0.6,
    beta=0.4,
)
agent = warehouse_rl.agents.OffPolicyAgent(
    loader_algorithm,
    picker_algorithm,
    loader_memory,
    picker_memory,
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


loader_eps_schedule: typing.Callable[[int], float] = exponential_annealing(
    1.0, 0.05, 0.994
)
loader_beta_schedule: typing.Callable[[int], float] = natural_exponential_annealing(
    0.4, 1.0, 0.01
)
picker_eps_schedule: typing.Callable[[int], float] = exponential_annealing(
    1.0, 0.05, 0.997
)
picker_beta_schedule: typing.Callable[[int], float] = natural_exponential_annealing(
    0.4, 1.0, 0.007
)
ckpt_loader = "ckpt/a"
os.makedirs(os.path.join(os.getcwd(), ckpt_loader), exist_ok=True)
ckpt_picker = "ckpt/b"
os.makedirs(os.path.join(os.getcwd(), ckpt_picker), exist_ok=True)


def train_fn(episode: int, step: int) -> None:
    agent.loader_algorithm.policy.set_eps_training(loader_eps_schedule(episode))
    if agent.loader_memory is not None:
        agent.loader_memory.set_beta(loader_beta_schedule(episode))
    agent.picker_algorithm.policy.set_eps_training(picker_eps_schedule(episode))
    if agent.picker_memory is not None:
        agent.picker_memory.set_beta(picker_beta_schedule(episode))


def save_last_fn() -> None:
    torch.save(
        agent.loader_algorithm.policy.state_dict(),
        os.path.join(ckpt_loader, "last.pth"),
    )
    torch.save(
        agent.loader_algorithm.optim.state_dict(),  # type: ignore
        os.path.join(ckpt_loader, "optim.pth"),
    )
    torch.save(
        agent.picker_algorithm.policy.state_dict(),
        os.path.join(ckpt_picker, "last.pth"),
    )
    torch.save(
        agent.picker_algorithm.optim.state_dict(),  # type: ignore
        os.path.join(ckpt_picker, "optim.pth"),
    )


def save_best_fn(episode: int) -> None:
    torch.save(
        agent.loader_algorithm.policy.state_dict(),
        os.path.join(ckpt_loader, "best.pth"),
    )
    torch.save(
        agent.picker_algorithm.policy.state_dict(),
        os.path.join(ckpt_picker, "best.pth"),
    )

train_env = tianshou.env.DummyVectorEnv(
    [
        lambda: warehouse_rl.warehouse.Warehouse(
            2, 2, 3, 3, True, 800, n_loaders, n_pickers, 20
        )
        for _ in range(16)
    ]
)
test_env = tianshou.env.DummyVectorEnv(
    [
        lambda: warehouse_rl.warehouse.Warehouse(
            2, 2, 3, 3, True, 800, n_loaders, n_pickers, 20
        )
        for _ in range(16)
    ]
)

trainer = warehouse_rl.agents.DecentralizedTrainer(
    batch_size=64,
    update_freq=200,
    test_freq=50,
    n_training_episodes=1000,
    n_testing_episodes=48,
    train_fn=train_fn,
    save_last_fn=save_last_fn,
    save_best_fn=save_best_fn,
)
# algorithm.to("cuda")
trainer.train(train_env, test_env, agent, n_loaders, n_pickers, True)
