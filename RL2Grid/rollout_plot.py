"""Roll out a PPO checkpoint for 1000 steps and plot states/actions over time."""
from __future__ import annotations

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

from env.utils import auxiliary_make_env
from alg.ppo.agent import Agent


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_expert(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint["args"]
    train_args.cuda = False
    train_args.n_envs = 1

    # Basic sanity on how the model was trained
    print(f"[ckpt] action_type={train_args.action_type}, norm_obs={getattr(train_args, 'norm_obs', None)}")

    def _make():
        env, _ = auxiliary_make_env(train_args, resume_run=False, test=True)
        return env

    env = gym.vector.SyncVectorEnv([_make])
    continuous = train_args.action_type == "redispatch"
    agent = Agent(env, train_args, continuous).to(_device())
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.eval()
    return agent, env


def collect_rollout(agent: Agent, env: gym.vector.SyncVectorEnv, steps: int):
    obs, _ = env.reset()
    obs_list = []
    act_list = []
    act_dim = env.single_action_space.shape[0]
    act_min = np.full(act_dim, np.inf, dtype=np.float64)
    act_max = np.full(act_dim, -np.inf, dtype=np.float64)
    act_sum = np.zeros(act_dim, dtype=np.float64)
    logit_min = np.full(act_dim, np.inf, dtype=np.float64)
    logit_max = np.full(act_dim, -np.inf, dtype=np.float64)
    logit_sum = np.zeros(act_dim, dtype=np.float64)
    t = 0
    while t < steps:
        with torch.no_grad():
            t_obs = torch.tensor(obs, dtype=torch.float32, device=_device())
            # Pre-sigmoid logits to detect saturation
            pre_sigmoid = agent.actor[:-1](t_obs).cpu().numpy()
            action = agent.get_eval_continuous_action(t_obs).cpu().numpy()
        if t < 3:  # log first few steps for debugging
            print(f"[step {t}] obs[:10]={obs[0][:10]}")
            print(f"[step {t}] logits[:10]={pre_sigmoid[0][:10]}")
            print(f"[step {t}] action[:10]={action[0][:10]}")
        act_min = np.minimum(act_min, action[0])
        act_max = np.maximum(act_max, action[0])
        act_sum += action[0]
        logit_min = np.minimum(logit_min, pre_sigmoid[0])
        logit_max = np.maximum(logit_max, pre_sigmoid[0])
        logit_sum += pre_sigmoid[0]
        obs_list.append(obs[0].copy())
        act_list.append(action[0].copy())
        obs, _, term, trunc, _ = env.step(action)
        done = np.logical_or(term, trunc)[0]
        if done:
            obs, _ = env.reset()
        t += 1
    act_mean = act_sum / steps
    logit_mean = logit_sum / steps
    print(f"[rollout] action min={act_min} max={act_max} mean={act_mean}")
    print(f"[rollout] logits min={logit_min} max={logit_max} mean={logit_mean}")
    return np.stack(obs_list), np.stack(act_list)


def plot_rollout(states: np.ndarray, actions: np.ndarray, out_path: str):
    steps = states.shape[0]
    max_state_show = min(10, states.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i in range(max_state_show):
        axes[0].plot(states[:, i], label=f"s{i}")
    axes[0].set_ylabel("State values")
    axes[0].set_title(f"States (first {max_state_show} dims)")
    axes[0].legend(ncol=5, fontsize=8)

    for j in range(actions.shape[1]):
        axes[1].plot(actions[:, j], label=f"a{j}")
    axes[1].set_ylabel("Action values")
    axes[1].set_xlabel("Step")
    axes[1].set_title("Actions")
    axes[1].legend(ncol=5, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path} (steps={steps}, states_dim={states.shape[1]}, actions_dim={actions.shape[1]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint/final_PPO_bus14_R_0_0__I__1766145995_42074.tar")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=str, default="rollout_plot.png")
    args = parser.parse_args()

    agent, env = load_expert(args.checkpoint)
    states, actions = collect_rollout(agent, env, steps=args.steps)
    plot_rollout(states, actions, args.output)


if __name__ == "__main__":
    main()
