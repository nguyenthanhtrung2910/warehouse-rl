import os

import numpy as np
import torch
from gymnasium import spaces
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.utils.net.common import Net

from warehouse_rl.agents import OffPolicyAgent

# from warehouse_rl.networks import Conv
from warehouse_rl.enums import RenderMode

# from warehouse_rl.warehouse import Warehouse
from warehouse_rl.warehouse_b import WarehouseB

n_agents = 4
net = Net(
    state_shape=19,
    action_shape=4,
    hidden_sizes=[1024, 1024, 512, 512, 256, 256, 128, 128,1 64, 64],
    norm_layer=torch.nn.LayerNorm,
    activation=torch.nn.ReLU,
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

env = WarehouseB(2, 2, 2, 2, True, 500, n_agents, 12, 4, render_mode=RenderMode.Human)
obs, _ = env.reset()

done = False
re = np.zeros(n_agents)
while not done:
    act_a = OffPolicyAgent.get_act(policy, obs.obs, obs.mask, False)
    next_obs, reward, termination, truncation, info = env.step(act_a)
    re += reward
    obs = next_obs
    done = termination or truncation
print(re.mean())
