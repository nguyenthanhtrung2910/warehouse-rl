import os

import gymnasium
import tianshou.algorithm.modelfree.dqn
import tianshou.utils.net.common
import torch

import warehouse_rl.agents_a
import warehouse_rl.enums
import warehouse_rl.warehouse_a

n_shuttles = 4
net = tianshou.utils.net.common.Net(
    state_shape=36 + 7,
    action_shape=4,
    hidden_sizes=[1024, 1024, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64],
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
    eps_training=0.0,
    eps_inference=0.0,
)
policy.load_state_dict(
    torch.load(
        os.path.join(os.getcwd(), "ckpt/endless", "best.pth"),
        weights_only=True,
    )
)
env = warehouse_rl.warehouse_a.Warehouse(
    2,
    2,
    3,
    3,
    True,
    1000,
    n_shuttles,
    20,
    render_mode=warehouse_rl.enums.RenderMode.Null,
    recording=True,
)
obs, _ = env.reset()
reward = 0
picker_reward = 0
done = False
while not done:
    act_a = warehouse_rl.agents_a.OffPolicyAgent.get_act(
        policy,
        obs.obs_a_o,
        obs.mask_a_ac,
        False,
    )
    next_obs, reward_a, termination, truncation, info = env.step(act_a)
    # print(f"Number parcels in line: {len(env.in_line_parcels)}")
    reward += reward_a.mean()
    obs = next_obs
    done: bool = termination or truncation
print(env.step_counter)
print(env.parcel_counter)
print(reward)
