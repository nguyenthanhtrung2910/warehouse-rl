import os

import torch
import numpy as np
from gymnasium import spaces
from tianshou.utils.net.common import Net

from warehouse_rl.train import DQNPolicyWithActionMask
from warehouse_rl.warehouse import Warehouse
from warehouse_rl.warehouse import RenderMode
from tianshou.data import Batch

FOLDER = "DQNagent"
os.makedirs(os.path.join(os.getcwd(), FOLDER), exist_ok=True)

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

env = Warehouse(2, 2, 2, 2, True, 700, RenderMode.Human)
obs, info = env.reset()

policy = DQNPolicyWithActionMask(
    model=net, action_space=spaces.Discrete(4)
)

policy.load_state_dict(torch.load(os.path.join(FOLDER, "last.pth"), weights_only=True))

done = False
while not done:
    info = np.array([info])
    obs = obs.reshape(1, -1)
    with torch.no_grad():
        act = policy(Batch(obs=obs, info=info)).act[0]
    obs, _, termination, truncation, info = env.step(act)
    done = termination or truncation
