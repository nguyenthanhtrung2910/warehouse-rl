import os
from typing import cast

import numpy as np
import torch
from gymnasium import spaces
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.data import Batch
from tianshou.data.types import ObsBatchProtocol

from warehouse_rl.networks import Conv
from warehouse_rl.enums import ObsMode
from warehouse_rl.warehouse import RenderMode, Warehouse

net = Conv(3)
policy = DiscreteQLearningPolicy(
    model=net, action_space=spaces.Discrete(4), eps_inference=1.0, eps_training=1.0
)
policy.load_state_dict(
    torch.load(
        os.path.join(os.getcwd(), "ckpt", "best.pth"),
        weights_only=True,
    )
)

env = Warehouse(2, 2, 2, 2, True, 500, RenderMode.Human, ObsMode.ResizedWindow)
obs, _ = env.reset()

done = False
re = 0
while not done:
    obs_batch = Batch(
        obs=Batch(obs=np.array([obs.obs]), mask=np.array([obs.mask])), info=None
    )
    obs_batch = cast(ObsBatchProtocol, obs_batch)
    with torch.no_grad():
        act = policy(obs_batch).act
    next_obs, reward, termination, truncation, info = env.step(act)
    re += reward
    obs = next_obs
    done = termination or truncation

print(re)
