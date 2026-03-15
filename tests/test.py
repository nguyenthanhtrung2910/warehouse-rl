import os
from typing import cast

import torch
from gymnasium import spaces
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.data import Batch
from tianshou.data.types import ObsBatchProtocol
from tianshou.utils.net.common import Net

from warehouse_rl.warehouse import RenderMode, Warehouse

net = Net(
    state_shape=3 + 16,
    action_shape=4,
    hidden_sizes=[1024, 1024, 512, 512, 256, 256, 128, 64],
    norm_layer=torch.nn.LayerNorm,
    dueling_param=(
        {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
        {"hidden_sizes": [32], "norm_layer": torch.nn.LayerNorm},
    ),
)
policy = DiscreteQLearningPolicy(
    model=net, action_space=spaces.Discrete(4), eps_inference=1.0, eps_training=1.0
)
policy.load_state_dict(
    torch.load(
        os.path.join(os.getcwd(), "ckpt", "best.pth"),
        weights_only=True,
    )
)
print(policy)
env = Warehouse(2, 2, 2, 2, True, 500, RenderMode.Human)
obs, _ = env.reset()

done = False
re = 0
while not done:
    obs_batch = Batch(
        obs=Batch(obs=obs.obs.reshape(1, -1), mask=obs.mask.reshape(1, -1)), info=None
    )
    obs_batch = cast(ObsBatchProtocol, obs_batch)
    with torch.no_grad():
        act = policy(obs_batch).act
        # act = policy.add_exploration_noise(act, obs_batch)[0]
    next_obs, reward, termination, truncation, info = env.step(act)
    # print(
    #     f"In step {env.n_steps}: observation {obs} action {act} next observation {next_obs} reward {reward}"
    # )
    re += reward
    obs = next_obs
    done = termination or truncation

print(re)
