"""Test SHAP stability under imperceptible Gaussian noise.

Experiment:
- Take a state s_t.
- Add tiny Gaussian noise to get s'_t such that the agent's action remains (nearly) unchanged.
- Compute SHAP values for both.
- If SHAP changes a lot despite action staying the same, explanations are unstable.

Runs this for N test samples and reports deltas plus a summary plot.
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

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from alg.ppo.agent import Agent
from env.utils import auxiliary_make_env
from common.utils import set_torch, set_random_seed


def collect_background(env, agent, n_samples: int) -> np.ndarray:
    obs_list = []
    obs, _ = env.reset()
    for _ in range(n_samples):
        obs_list.append(obs[0])
        with th.no_grad():
            t_obs = th.tensor(obs, dtype=th.float32)
            action = agent.get_eval_continuous_action(t_obs)
        obs, _, term, trunc, _ = env.step(action.detach().cpu().numpy())
        if np.logical_or(term, trunc)[0]:
            obs, _ = env.reset()
    return np.stack(obs_list)


def get_shap_array(values) -> np.ndarray:
    """Normalize SHAP results to (samples, features, outputs)."""
    if isinstance(values, list):
        arr = np.stack(values, axis=-1)
    else:
        arr = values
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected SHAP shape {arr.shape}")
    return arr


def ensure_small_action_change(agent: Agent, state: np.ndarray, noise_sigma: float, tol: float, max_tries: int = 5) -> Tuple[np.ndarray, float]:
    """Add isotropic noise and shrink until action change is below tolerance."""
    sigma = noise_sigma
    base = th.tensor(state, dtype=th.float32).unsqueeze(0)
    with th.no_grad():
        mu_base = agent.actor(base).squeeze(0)
    for _ in range(max_tries):
        noise = np.random.normal(scale=sigma, size=state.shape).astype(np.float32)
        noisy = state + noise
        with th.no_grad():
            mu_noisy = agent.actor(th.tensor(noisy, dtype=th.float32).unsqueeze(0)).squeeze(0)
        delta = th.norm(mu_noisy - mu_base, p=2).item()
        if delta <= tol:
            return noisy, delta
        sigma *= 0.5  # reduce noise and retry
    return noisy, delta  # best effort


def main():
    parser = argparse.ArgumentParser(description="SHAP stability under tiny noise")
    parser.add_argument("--run-name", required=True, help="Checkpoint run name (without .tar)")
    parser.add_argument("--n-samples", type=int, default=256, help="Background samples for SHAP")
    parser.add_argument("--n-test", type=int, default=20, help="Number of clean test states")
    parser.add_argument("--n-noisy", type=int, default=20, help="Noisy perturbations per test state")
    parser.add_argument("--noise-sigma", type=float, default=1e-3, help="Std of Gaussian noise before scaling down")
    parser.add_argument("--action-tol", type=float, default=1e-4, help="Max L2 change in action mean allowed")
    parser.add_argument("--seed", type=int, default=24, help="Random seed")
    parser.add_argument("--output", type=str, default="shap_noise_stability.png", help="Output plot path")
    args, _ = parser.parse_known_args()

    set_random_seed(args.seed)
    device = set_torch(n_threads=1)

    checkpoint_path = f"RL2Grid/checkpoint/{args.run_name}.tar"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"checkpoint/{args.run_name}.tar"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

    ckpt = th.load(checkpoint_path, map_location="cpu")
    train_args = ckpt["args"]
    train_args.cuda = False
    train_args.n_envs = 1
    
    def make_env():
        env, _ = auxiliary_make_env(train_args, resume_run=False, test=True)
        return env

    env = gym.vector.SyncVectorEnv([make_env])
    agent = Agent(env, train_args, continuous_actions=True).to(device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.eval()

    # Background data for SHAP
    print(f"Collecting {args.n_samples} background samples...")
    background = collect_background(env, agent, args.n_samples)
    background_tensor = th.tensor(background, dtype=th.float32, device=device)
    background_std = background.std(axis=0)
    print(f"Background shape: {background.shape}; obs space shape: {env.single_observation_space.shape}")

    # Build SHAP explainer
    explainer = shap.DeepExplainer(agent.actor, background_tensor)

    # Collect test states along a rollout
    test_states = []
    obs, _ = env.reset()
    while len(test_states) < args.n_test:
        test_states.append(obs[0].copy())
        with th.no_grad():
            t_obs = th.tensor(obs, dtype=th.float32)
            action = agent.get_eval_continuous_action(t_obs)
        obs, _, term, trunc, _ = env.step(action.detach().cpu().numpy())
        if np.logical_or(term, trunc)[0]:
            obs, _ = env.reset()

    # Build clean and noisy batches
    clean_states = []
    grouped_noisy_states: List[np.ndarray] = []
    action_deltas: List[List[float]] = []
    for state_idx, state in enumerate(test_states):
        per_noisy = []
        per_deltas = []
        while len(per_noisy) < args.n_noisy:
            noisy_state, act_delta = ensure_small_action_change(
                agent, state, args.noise_sigma, args.action_tol
            )
            per_noisy.append(noisy_state)
            per_deltas.append(act_delta)
        clean_states.append(state)
        grouped_noisy_states.append(np.stack(per_noisy))
        action_deltas.append(per_deltas)

    clean_batch = th.tensor(np.stack(clean_states), dtype=th.float32, device=device)
    noisy_flat = np.concatenate(grouped_noisy_states, axis=0)
    noisy_batch = th.tensor(noisy_flat, dtype=th.float32, device=device)

    # SHAP on batches (converted to samples x features x outputs)
    shap_clean = get_shap_array(explainer.shap_values(clean_batch, check_additivity=False))
    shap_noisy = get_shap_array(explainer.shap_values(noisy_batch, check_additivity=False))
    print(f"SHAP clean shape (samples, features, outputs): {shap_clean.shape}")
    print(f"SHAP noisy shape (samples, features, outputs): {shap_noisy.shape}")

    n_clean, n_features, n_outputs = shap_clean.shape
    expected_feat = clean_batch.shape[1]
    if n_features != expected_feat:
        print(
            f"Warning: SHAP features ({n_features}) != state size ({expected_feat}). "
            "Check env observation config / feature list."
        )

    n_noisy_total = shap_noisy.shape[0]
    if n_noisy_total != n_clean * args.n_noisy:
        raise RuntimeError(
            f"Expected {n_clean * args.n_noisy} noisy samples, got {n_noisy_total}."
        )

    shap_noisy = shap_noisy.reshape(n_clean, args.n_noisy, n_features, n_outputs)

    mean_rel_diffs = []
    mean_rel_max_diffs = []
    for idx in range(n_clean):
        clean_i = shap_clean[idx]
        noisy_i = shap_noisy[idx]  # (n_noisy, features, outputs)
        diffs = np.abs(noisy_i - clean_i[None, :, :])
        eps = 1e-9
        rel_diffs = diffs / (np.abs(clean_i)[None, :, :] + eps)
        per_noise_mean = rel_diffs.reshape(args.n_noisy, -1).mean(axis=1)
        per_noise_max = rel_diffs.reshape(args.n_noisy, -1).max(axis=1)
        mean_rel = per_noise_mean.mean() * 100.0
        mean_rel_max = per_noise_max.mean() * 100.0
        mean_rel_diffs.append(mean_rel)
        mean_rel_max_diffs.append(mean_rel_max)

        action_stats = np.array(action_deltas[idx])
        print(
            f"State {idx+1:02d}: mean action Δ={action_stats.mean():.2e}, max action Δ={action_stats.max():.2e}, "
            f"mean SHAP Δ={mean_rel:.2f}%, mean max SHAP Δ={mean_rel_max:.2f}%"
        )

    mean_diffs_arr = np.array(mean_rel_diffs)
    mean_max_diffs_arr = np.array(mean_rel_max_diffs)
    print("\nAggregate over test states:")
    print(
        f"  mean(|ΔSHAP|) %: {mean_diffs_arr.mean():.2f}% ± {mean_diffs_arr.std():.2f}%"
    )
    print(
        f"  mean(max |ΔSHAP|) %: {mean_max_diffs_arr.mean():.2f}% ± {mean_max_diffs_arr.std():.2f}%"
    )

    # Plot summary (separate figures, saved as PDF)
    x = np.arange(1, len(mean_diffs_arr) + 1, dtype=int)
    base, _ = os.path.splitext(args.output)

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.bar(x, mean_diffs_arr, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Test sample index")
    ax1.set_ylabel("Mean relative difference (%)")
    ax1.set_xticks(x)
    fig1.tight_layout()
    out1 = f"{base}_mean.pdf"
    fig1.savefig(out1, bbox_inches="tight", format="pdf")
    print(f"Saved plot to {out1}")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.bar(x, mean_max_diffs_arr, color="indianred", alpha=0.8)
    ax2.set_xlabel("Test sample index")
    ax2.set_ylabel("Mean of per-noise max diff (%)")
    ax2.set_xticks(x)
    fig2.tight_layout()
    out2 = f"{base}_mean_max.pdf"
    fig2.savefig(out2, bbox_inches="tight", format="pdf")
    print(f"Saved plot to {out2}")


if __name__ == "__main__":
    main()
