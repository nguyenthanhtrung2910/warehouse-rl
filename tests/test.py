import os

import gymnasium
import tianshou.algorithm.modelfree.dqn
import tianshou.utils.net.common
import torch

import warehouse_rl.agents
import warehouse_rl.enums
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
    model=net_loader,
    action_space=gymnasium.spaces.Discrete(4),
    eps_training=1.0,
    eps_inference=1.0,
)
picker_policy: tianshou.algorithm.modelfree.dqn.DiscreteQLearningPolicy[
    tianshou.utils.net.common.Net
] = tianshou.algorithm.modelfree.dqn.DiscreteQLearningPolicy(
    model=net_picker,
    action_space=gymnasium.spaces.Discrete(4),
    eps_training=1.0,
    eps_inference=1.0,
)
loader_policy.load_state_dict(
    torch.load(
        os.path.join(os.getcwd(), "ckpt/a", "best.pth"),
        weights_only=True,
    )
)
picker_policy.load_state_dict(
    torch.load(
        os.path.join(os.getcwd(), "ckpt/b", "best.pth"),
        weights_only=True,
    )
)
env = warehouse_rl.warehouse.Warehouse(
    2,
    2,
    3,
    3,
    True,
    800,
    n_loaders,
    n_pickers,
    request_freq=20,
    render_mode=warehouse_rl.enums.RenderMode.Null,
    recording=True,
)
obs, _ = env.reset()
loader_reward = 0
picker_reward = 0
done = False
while not done:
    act_a = warehouse_rl.agents.OffPolicyAgent.get_act(
        loader_policy,
        picker_policy,
        obs.loader_obs_a_o,
        obs.loader_mask_a_ac,
        obs.picker_obs_a_o,
        obs.picker_mask_a_ac,
        False,
    )
    next_obs, reward, termination, truncation, info = env.step(act_a)
    loader_reward += reward[:n_loaders].mean()
    picker_reward += reward[n_loaders:].mean()
    obs = next_obs
    done: bool = termination or truncation
print(f"loader: {loader_reward}, picker {picker_reward}")
