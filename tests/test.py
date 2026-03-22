import os

import gymnasium
import numpy as np
import tianshou.algorithm.modelfree.dqn
import tianshou.utils.net.common
import torch

import warehouse_rl.agents
import warehouse_rl.enums
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
    model=net,
    action_space=gymnasium.spaces.Discrete(4),
    eps_inference=1.0,
    eps_training=1.0,
)
policy.load_state_dict(
    torch.load(
        os.path.join(os.getcwd(), "ckpt/b", "best.pth"),
        weights_only=True,
    )
)
env = warehouse_rl.warehouse_b.WarehouseB(
    2,
    2,
    3,
    3,
    True,
    n_steps=500,
    n_shuttles=n_agents,
    n_parcels=20,
    n_requested=6,
    render_mode=warehouse_rl.enums.RenderMode.Null,
    recording=True,
)
obs, _ = env.reset()
done = False
re: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(n_agents)
while not done:
    act_a = warehouse_rl.agents.OffPolicyAgent.get_act(policy, obs.obs, obs.mask, False)
    next_obs, reward, termination, truncation, info = env.step(act_a)
    re += reward
    obs = next_obs
    done: bool = termination or truncation
env.close()
print(re.mean())
