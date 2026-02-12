"""Executable script to train counterfactual modules and compute dispatch adjustments.

This script loads a frozen PPO checkpoint, trains the counterfactual encoder stack on
fresh trajectories, and then generates counterfactual redispatch states for scenarios
with line overloads. A simple weighted-L1 optimization is performed to obtain the
minimum-cost dispatch vector that adheres to the counterfactual guidance while staying
close to the operator's planned dispatch profile.
"""
from __future__ import annotations

import argparse
import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from alg.ppo.agent import Agent
from counterfactual import (
    CounterfactualTrainer,
    Discriminator,
    Encoder,
    Generator,
    WassersteinAE,
    generate_counterfactual,
)
from env.utils import auxiliary_make_env, load_config


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feature_names(g2op_env, args, config) -> Tuple[List[str], Dict[str, List[int]]]:
    env_id = args.env_id
    env_config = config['environments']
    state_attrs = config['state_attrs']

    obs_attrs = list(state_attrs['default'])
    if env_config[env_id]['maintenance']:
        obs_attrs += state_attrs['maintenance']

    env_type = args.action_type.lower()
    if env_type == 'topology':
        obs_attrs += state_attrs['topology']
    else:
        obs_attrs += state_attrs['redispatch']
        if env_config[env_id]['renewable']:
            obs_attrs += state_attrs['curtailment']
        if env_config[env_id]['battery']:
            obs_attrs += state_attrs['storage']

    feature_names: List[str] = []
    feature_index: Dict[str, List[int]] = {}
    for attr in obs_attrs:
        if attr in ['gen_p', 'gen_q', 'gen_v', 'gen_theta', 'target_dispatch', 'actual_dispatch', 'gen_margin_up', 'gen_margin_down', 'gen_p_before_curtail', 'curtailment', 'curtailment_limit']:
            names = [f"{attr}_{name}" for name in g2op_env.name_gen]
        elif attr in ['load_p', 'load_q', 'load_v', 'load_theta']:
            names = [f"{attr}_{name}" for name in g2op_env.name_load]
        elif attr in ['rho', 'line_status', 'timestep_overflow', 'time_before_cooldown_line', 'time_next_maintenance', 'duration_next_maintenance', 'p_or', 'q_or', 'v_or', 'a_or', 'theta_or', 'p_ex', 'q_ex', 'v_ex', 'a_ex', 'theta_ex']:
            names = [f"{attr}_{name}" for name in g2op_env.name_line]
        elif attr == 'topo_vect':
            names = [f"topo_vect_{i}" for i in range(g2op_env.dim_topo)]
        elif attr == 'time_before_cooldown_sub':
            names = [f"time_before_cooldown_sub_{i}" for i in range(g2op_env.n_sub)]
        elif attr in ['storage_charge', 'storage_power_target', 'storage_power', 'storage_theta']:
            names = [f"{attr}_{i}" for i in range(g2op_env.n_storage)]
        else:
            names = [attr]
        feature_names.extend(names)
        start = len(feature_names) - len(names)
        indices = list(range(start, start + len(names)))
        feature_index.setdefault(attr, []).extend(indices)
    return feature_names, feature_index


def _extract_feature(obs: np.ndarray, indices: List[int]) -> np.ndarray:
    if not indices:
        raise ValueError("Requested feature slice is empty.")
    return obs[indices]


def _collect_rollouts(
    env: gym.vector.SyncVectorEnv,
    agent: Agent,
    n_samples: int,
    overload_threshold: float,
    rho_indices: List[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    observations: List[np.ndarray] = []
    overload_states: List[np.ndarray] = []
    safe_actions: List[np.ndarray] = []

    obs, _ = env.reset()
    needed = n_samples
    while needed > 0 or len(overload_states) < max(1, n_samples // 10):
        state = obs[0].copy()
        observations.append(state)
        needed -= 1

        with torch.no_grad():
            t_obs = torch.tensor(obs, dtype=torch.float32)
            action = agent.get_eval_continuous_action(t_obs).cpu().numpy()

        next_obs, _, term, trunc, info = env.step(action)
        next_rho = next_obs[0][rho_indices]
        if next_rho.max() > overload_threshold:
            overload_states.append(state.copy())
        else:
            safe_actions.append(action[0].copy())

        obs = next_obs
        done = np.logical_or(term, trunc)
        if done[0]:
            obs, _ = env.reset()
        if len(observations) >= n_samples and len(overload_states) >= max(1, n_samples // 10):
            break

    if not safe_actions:
        safe_actions.append(np.zeros_like(action[0]))
    return observations, overload_states, safe_actions


def _build_dataloader(states: Sequence[np.ndarray], batch_size: int) -> DataLoader:
    tensor = torch.tensor(np.stack(states), dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _safe_action_template(actions: Sequence[np.ndarray]) -> torch.Tensor:
    stacked = np.stack(actions)
    return torch.tensor(stacked.mean(axis=0), dtype=torch.float32)


def _extract_dispatch_vector(obs: np.ndarray, feature_index: Dict[str, List[int]], attr: str) -> np.ndarray:
    if attr in feature_index:
        return obs[feature_index[attr]]
    raise KeyError(f"Attribute {attr} not found in feature names.")


def load_agent_from_checkpoint(checkpoint_path: str) -> Tuple[Agent, gym.vector.SyncVectorEnv, argparse.Namespace, torch.device]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint['args']
    train_args.cuda = False
    train_args.n_envs = 1

    def make_env():
        env, _ = auxiliary_make_env(train_args, resume_run=False, test=True)
        return env

    env = gym.vector.SyncVectorEnv([make_env])
    continuous = train_args.action_type == "redispatch"
    agent = Agent(env, train_args, continuous).to(_device())
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    agent.eval()
    return agent, env, train_args, _device()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train counterfactual modules and compute redispatch adjustments.")
    parser.add_argument("--checkpoint", required=True, help="Path to PPO checkpoint (.tar)")
    parser.add_argument("--n-train-samples", type=int, default=2048, help="Number of rollout samples for training")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encoding-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--overload-threshold", type=float, default=1.02)
    parser.add_argument("--max-overload-cases", type=int, default=5)
    parser.add_argument("--n-noisy", type=int, default=10, help="Number of noisy CF samples per base counterfactual")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Std dev of Gaussian noise added to state")
    parser.add_argument(
        "--comparison-file",
        type=str,
        default="counterfactual_comparison.csv",
        help="Path to write the original vs counterfactual state table (CSV).",
    )
    parser.add_argument(
        "--diff-plot",
        type=str,
        default="counterfactual_diff.pdf",
        help="Path to save per-case noisy CF robustness plot.",
    )
    args = parser.parse_args()

    agent, env, train_args, device = load_agent_from_checkpoint(args.checkpoint)
    base_env = env.envs[0]
    g2op_env = base_env.init_env
    config = load_config(train_args.env_config_path)
    feature_names, feature_index = get_feature_names(g2op_env, train_args, config)
    dispatch_attr = 'target_dispatch'
    print("Using 'target_dispatch' for dispatch extraction.")

    # Limits for validation
    gen_pmin = g2op_env.gen_pmin
    gen_pmax = g2op_env.gen_pmax

    rho_indices = feature_index.get('rho', [])
    if not rho_indices:
        raise RuntimeError("Could not locate rho entries in observation vector.")

    observations, overload_states, safe_actions = _collect_rollouts(
        env,
        agent,
        args.n_train_samples,
        args.overload_threshold,
        rho_indices,
    )

    dataloader = _build_dataloader(observations, args.batch_size)
    obs_dim = env.single_observation_space.shape[0]
    action_dim = int(np.prod(env.single_action_space.shape))

    encoder = Encoder(obs_dim, args.encoding_dim)
    generator = Generator(args.encoding_dim, action_dim, obs_dim)
    discriminator = Discriminator(args.encoding_dim, action_dim)
    wae = WassersteinAE(latent_dim=env.single_observation_space.shape[0], projection_dim=args.projection_dim)

    trainer = CounterfactualTrainer(agent, dataloader, encoder, generator, discriminator, wae, device=device)
    for epoch in range(args.train_epochs):
        metrics = trainer.train_epoch()
        print(f"Epoch {epoch+1}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    mean_diff_series: List[float] = []
    max_diff_series: List[float] = []

    with open(args.comparison_file, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "case_id",
            "feature",
            "original",
            "counterfactual",
            "delta",
            "fault_flag",
            "cf_rho_violation",
        ])

        for idx, state in enumerate(overload_states[: args.max_overload_cases]):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            state_np = state_tensor.cpu().numpy().reshape(-1)
            planned_dispatch = _extract_dispatch_vector(state_np, feature_index, dispatch_attr)

            target_action = torch.zeros(action_dim, dtype=torch.float32, device=device)

            cf_state = generate_counterfactual(
                state_tensor,
                target_action,
                agent,
                encoder,
                generator,
                wae,
                device=device,
            )

            cf_np = cf_state.detach().cpu().numpy().reshape(-1)
            cf_dispatch = _extract_dispatch_vector(cf_np, feature_index, dispatch_attr)

            print("-" * 60)
            print(f"Case {idx+1}:")
            cf_rho = _extract_dispatch_vector(cf_np, feature_index, 'rho') if 'rho' in feature_index else np.array([])
            safe = cf_rho.size == 0 or cf_rho.max() < 1.0

            print(f"  Planned dispatch: {np.round(planned_dispatch, 3)}")
            print(f"  Counterfactual dispatch: {np.round(cf_dispatch, 3)}")
            status = "SAFE" if safe else "WARNING: rho>=1 detected"
            print(f"  Grid safety status: {status}")

            # Check generation limits
            if train_args.norm_obs:
                print("  [NOTE] Observations normalized; raw value comparison to MW limits may be scale-mismatched.")
            
            if len(cf_dispatch) == len(gen_pmin):
                min_vio = cf_dispatch < gen_pmin - 1e-3
                max_vio = cf_dispatch > gen_pmax + 1e-3
                if np.any(min_vio | max_vio):
                    print("  WARNING: Dispatch out of generation bounds (gen_p_min / gen_p_max):")
                    for i in np.where(min_vio | max_vio)[0]:
                        viol_val = cf_dispatch[i]
                        bounds = (gen_pmin[i], gen_pmax[i])
                        print(f"    Gen {i}: {viol_val:.3f} not in {bounds}")
            else:
                print(f"  (Skipping bounds check: dispatch dim {len(cf_dispatch)} != n_gen {len(gen_pmin)})")

            # Persist original vs counterfactual state vector
            for f_idx, feature in enumerate(feature_names):
                orig_val = float(state_np[f_idx])
                cf_val = float(cf_np[f_idx])
                delta = cf_val - orig_val
                fault_flag = (
                    f_idx in rho_indices and (orig_val > args.overload_threshold or cf_val > args.overload_threshold)
                )
                cf_violation = f_idx in rho_indices and cf_val > args.overload_threshold
                writer.writerow([
                    idx + 1,
                    feature,
                    orig_val,
                    cf_val,
                    delta,
                    int(fault_flag),
                    int(cf_violation),
                ])

            # Noisy counterfactual robustness: perturb state, regenerate CF, and report diffs
            if args.n_noisy > 0:
                noise_std = float(args.noise_std)
                diffs = []
                for j in range(args.n_noisy):
                    noise = torch.randn_like(state_tensor) * noise_std
                    noisy_state = state_tensor + noise
                    cf_noisy = generate_counterfactual(
                        noisy_state,
                        target_action,
                        agent,
                        encoder,
                        generator,
                        wae,
                        device=device,
                    )
                    cf_noisy_np = cf_noisy.detach().cpu().numpy().reshape(-1)
                    diffs.append(cf_noisy_np - cf_np)

                diffs_np = np.stack(diffs)
                mean_diff = diffs_np.mean(axis=0)
                max_abs_diff = np.abs(diffs_np).max(axis=0)
                print("  Noisy CF robustness (noise std={:.4f}, {} samples):".format(noise_std, args.n_noisy))
                print("    mean_diff (L1 summary): {:.4f}".format(np.mean(np.abs(mean_diff))))
                print("    max_abs_diff (L∞ summary): {:.4f}".format(np.max(max_abs_diff)))
                mean_diff_series.append(float(np.mean(np.abs(mean_diff))))
                max_diff_series.append(float(np.max(max_abs_diff)))

    if mean_diff_series:
        cases = np.arange(1, len(mean_diff_series) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(cases, mean_diff_series, marker="o", label="mean_diff")
        plt.plot(cases, max_diff_series, marker="s", label="max_abs_diff")
        plt.xlabel("Sample index")
        plt.ylabel("Diff magnitude")
        plt.title("Noisy CF robustness per sample")
        plt.grid(True, alpha=0.3)
        plt.xticks(cases)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.diff_plot, dpi=200, format="pdf")
        print(f"Diff plot saved to {args.diff_plot}")

    print(f"Comparison table written to {args.comparison_file}")


if __name__ == "__main__":
    main()
