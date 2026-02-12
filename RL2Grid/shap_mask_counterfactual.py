"""SHAP-based masking test for PPO counterfactual importance.

Steps:
1) Load PPO checkpoint and Grid2Op env.
2) Collect background observations for SHAP.
3) Pick one test state, compute SHAP values, and select top-k features.
4) Mask those features with a baseline (background mean).
5) Compare policy output on original vs masked state (log-prob of original action and mean action delta).
6) Plot SHAP importances and effect metrics.
"""
import argparse
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch as th
import gymnasium as gym

# Add script directory to path so imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from alg.ppo.agent import Agent
from env.utils import auxiliary_make_env, load_config
from common.utils import set_torch, set_random_seed
from explain_shap import get_feature_names  # reuse helper


def collect_rollout_background(env, agent, n_samples: int) -> np.ndarray:
    """Collect background observations using the frozen agent policy."""
    background_obs = []
    obs, _ = env.reset()
    for _ in range(n_samples):
        background_obs.append(obs[0])
        with th.no_grad():
            t_obs = th.tensor(obs).float()
            action = agent.get_eval_continuous_action(t_obs)
        obs, _, term, trunc, _ = env.step(action.detach().cpu().numpy())
        if np.logical_or(term, trunc)[0]:
            obs, _ = env.reset()
    return np.stack(background_obs)


def collect_test_states(env, agent, n_test: int) -> np.ndarray:
    """Grab a sequence of test observations (deterministic given seed)."""
    test_obs = []
    obs, _ = env.reset()
    for _ in range(n_test):
        test_obs.append(obs[0])
        with th.no_grad():
            t_obs = th.tensor(obs).float()
            action = agent.get_eval_continuous_action(t_obs)
        obs, _, term, trunc, _ = env.step(action.detach().cpu().numpy())
        if np.logical_or(term, trunc)[0]:
            obs, _ = env.reset()
    return np.stack(test_obs)


def main():
    parser = argparse.ArgumentParser(description="SHAP masking counterfactual test")
    parser.add_argument("--run-name", required=True, help="Checkpoint run name (without .tar)")
    parser.add_argument("--k", type=int, default=20, help="Top-k features to mask")
    parser.add_argument("--n-samples", type=int, default=256, help="Background samples for SHAP")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default="shap_mask_test.png", help="Output plot path")
    parser.add_argument("--n-test", type=int, default=5, help="Number of test states to evaluate")
    args, _ = parser.parse_known_args()

    set_random_seed(args.seed)
    # set_num_threads requires positive int; use 1 for safety in this standalone script.
    device = set_torch(n_threads=1)

    checkpoint_path = f"RL2Grid/checkpoint/{args.run_name}.tar"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"checkpoint/{args.run_name}.tar"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = th.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint["args"]
    train_args.cuda = False
    train_args.n_envs = 1
    if train_args.action_type != "redispatch":
        raise ValueError("This script supports only redispatch (continuous) agents.")

    # Build env
    def make_env():
        env, _ = auxiliary_make_env(train_args, resume_run=False, test=True)
        return env

    env = gym.vector.SyncVectorEnv([make_env])
    agent = Agent(env, train_args, continuous_actions=True).to(device)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.eval()

    # Collect background
    print(f"Collecting {args.n_samples} background samples...")
    background_data = collect_rollout_background(env, agent, args.n_samples)
    background_tensor = th.tensor(background_data, dtype=th.float32, device=device)
    baseline = background_data.mean(axis=0)
    # Test states to evaluate
    print(f"Collecting {args.n_test} test states...")
    test_states = collect_test_states(env, agent, args.n_test)
    explainer = shap.DeepExplainer(agent.actor, background_tensor)

    # Feature names (once)
    feature_names = None
    try:
        gym_env = env.envs[0]
        g2op_env = gym_env.init_env
        config = load_config(train_args.env_config_path)
        # Infer number of features from background
        n_features_from_data = background_data.shape[1]
        feature_names = get_feature_names(g2op_env, train_args, config)
        if len(feature_names) != n_features_from_data:
            feature_names = None
    except Exception as e:
        print(f"Warning: could not fetch feature names: {e}")
        feature_names = None

    logp_drops: List[float] = []
    action_deltas: List[float] = []

    for i, state in enumerate(test_states):
        state_tensor = th.tensor(state, dtype=th.float32, device=device).unsqueeze(0)
        shap_values_raw = explainer.shap_values(state_tensor, check_additivity=False)
        shap_array = np.array(shap_values_raw)
        if shap_array.ndim != 3:
            raise RuntimeError(f"Unexpected SHAP shape {shap_array.shape}, expected (outputs, samples, features)")

        _, n_features, _ = shap_array.shape
        shap_vals = shap_array[0, :, :]  # (features, outputs)
        shap_importance = np.abs(shap_vals).sum(axis=1)

        k = min(args.k, n_features)
        top_idx = np.argsort(shap_importance)[::-1][:k]

        masked_state = state.copy()
        masked_state[top_idx] = baseline[top_idx] +10
        masked_tensor = th.tensor(masked_state, dtype=th.float32, device=device).unsqueeze(0)

        with th.no_grad():
            mu_orig = agent.actor(state_tensor).squeeze(0)
            mu_mask = agent.actor(masked_tensor).squeeze(0)
            logstd = agent.logstd.to(device).expand_as(mu_orig.unsqueeze(0)).squeeze(0)
            dist_orig = th.distributions.Normal(mu_orig, th.exp(logstd))
            dist_mask = th.distributions.Normal(mu_mask, th.exp(logstd))
            logp_orig = dist_orig.log_prob(mu_orig).sum().item()
            logp_mask_on_orig = dist_mask.log_prob(mu_orig).sum().item()

        delta_action = th.norm(mu_mask - mu_orig, p=2).item()
        logp_drop = logp_mask_on_orig - logp_orig

        logp_drops.append(logp_drop)
        action_deltas.append(delta_action)

        print(f"\nState {i+1}/{args.n_test}:")
        print(f"  Log p(orig|orig)   : {logp_orig:.3f}")
        print(f"  Log p(orig|masked) : {logp_mask_on_orig:.3f}")
        print(f"  Δlogp              : {logp_drop:.3f}")
        print(f"  Δ||μ||2            : {delta_action:.3f}")
        print("  Top-k features masked:")
        for rank, idx in enumerate(top_idx, start=1):
            name = feature_names[idx] if feature_names else f"f{idx}"
            print(f"    {rank}. {name}: SHAP={shap_importance[idx]:.4f}")

    # Aggregate summary
    logp_drops_arr = np.array(logp_drops)
    action_deltas_arr = np.array(action_deltas)
    print("\nAggregate over test states:")
    print(f"  Δlogp mean/std : {logp_drops_arr.mean():.3f} / {logp_drops_arr.std():.3f}")
    print(f"  Δ||μ||2 mean/std: {action_deltas_arr.mean():.3f} / {action_deltas_arr.std():.3f}")

    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(logp_drops_arr, bins=min(10, len(logp_drops_arr)), color="indianred", alpha=0.8)
    axes[0].set_title("Δlogp (orig action under masked)")
    axes[0].axvline(logp_drops_arr.mean(), color="black", linestyle="--", label=f"mean={logp_drops_arr.mean():.2f}")
    axes[0].legend()

    axes[1].hist(action_deltas_arr, bins=min(10, len(action_deltas_arr)), color="steelblue", alpha=0.8)
    axes[1].set_title("L2 delta in action means")
    axes[1].axvline(action_deltas_arr.mean(), color="black", linestyle="--", label=f"mean={action_deltas_arr.mean():.2f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
