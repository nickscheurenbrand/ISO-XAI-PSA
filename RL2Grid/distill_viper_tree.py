"""Distill a PPO expert into a Decision Tree via a simple VIPER/DAGGER-style loop.

- Expert: PPO checkpoint (torch .tar) using RL2Grid redispatch task (bus14).
- Student: sklearn DecisionTreeRegressor (multi-output for continuous redispatch).
- Loop: student acts (expert only first iter), collect obs, relabel with expert, weight by line margin criticalness, aggregate, fit tree.
- Eval: run one episode with the final student and report total reward.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor
import gymnasium as gym

from env.utils import auxiliary_make_env, load_config
from env.config import get_env_args
from alg.ppo.config import get_alg_args
from alg.ppo.agent import Agent


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(env_id: str, seed: int = 0):
    saved_env_args = argparse.Namespace(**vars(get_env_args()), **vars(get_alg_args()))
    saved_env_args.env_id = env_id
    saved_env_args.action_type = "redispatch"
    saved_env_args.n_envs = 1
    saved_env_args.cuda = False
    saved_env_args.seed = seed
    env, _ = auxiliary_make_env(saved_env_args, resume_run=False, test=True)
    return env, saved_env_args


def load_expert(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint["args"]
    train_args.cuda = False
    train_args.n_envs = 1

    def _make():
        env, _ = auxiliary_make_env(train_args, resume_run=False, test=True)
        return env

    env = gym.vector.SyncVectorEnv([_make])
    continuous = train_args.action_type == "redispatch"
    agent = Agent(env, train_args, continuous).to(_device())
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.eval()
    return agent, train_args


def get_feature_indices(g2op_env, train_args) -> dict:
    config = load_config(train_args.env_config_path)
    env_config = config["environments"][train_args.env_id]
    state_attrs = config["state_attrs"]

    obs_attrs = list(state_attrs["default"])
    if env_config.get("maintenance"):
        obs_attrs += state_attrs.get("maintenance", [])
    if train_args.action_type == "topology":
        obs_attrs += state_attrs.get("topology", [])
    else:
        obs_attrs += state_attrs.get("redispatch", [])
        if env_config.get("renewable"):
            obs_attrs += state_attrs.get("curtailment", [])
        if env_config.get("battery"):
            obs_attrs += state_attrs.get("storage", [])

    feature_index = {}
    feature_names = []
    for attr in obs_attrs:
        if attr in [
            "gen_p",
            "gen_q",
            "gen_v",
            "gen_theta",
            "target_dispatch",
            "actual_dispatch",
            "gen_margin_up",
            "gen_margin_down",
            "gen_p_before_curtail",
            "curtailment",
            "curtailment_limit",
        ]:
            names = [f"{attr}_{name}" for name in g2op_env.name_gen]
        elif attr in ["load_p", "load_q", "load_v", "load_theta"]:
            names = [f"{attr}_{name}" for name in g2op_env.name_load]
        elif attr in [
            "rho",
            "line_status",
            "timestep_overflow",
            "time_before_cooldown_line",
            "time_next_maintenance",
            "duration_next_maintenance",
            "p_or",
            "q_or",
            "v_or",
            "a_or",
            "theta_or",
            "p_ex",
            "q_ex",
            "v_ex",
            "a_ex",
            "theta_ex",
        ]:
            names = [f"{attr}_{name}" for name in g2op_env.name_line]
        elif attr == "topo_vect":
            names = [f"topo_vect_{i}" for i in range(g2op_env.dim_topo)]
        elif attr == "time_before_cooldown_sub":
            names = [f"time_before_cooldown_sub_{i}" for i in range(g2op_env.n_sub)]
        elif attr in ["storage_charge", "storage_power_target", "storage_power", "storage_theta"]:
            names = [f"{attr}_{i}" for i in range(g2op_env.n_storage)]
        else:
            names = [attr]
        start = len(feature_names)
        feature_names.extend(names)
        idxs = list(range(start, start + len(names)))
        feature_index[attr] = feature_index.get(attr, []) + idxs
    return feature_index


def expert_action(agent: Agent, obs: np.ndarray) -> np.ndarray:
    # obs shape (obs_dim,)
    with torch.no_grad():
        t_obs = torch.tensor(obs[None, :], dtype=torch.float32, device=_device())
        action = agent.get_eval_continuous_action(t_obs).cpu().numpy()[0]
    return action


def student_action(student: DecisionTreeRegressor | None, obs: np.ndarray, action_dim: int) -> np.ndarray:
    if student is None:
        return np.zeros(action_dim, dtype=np.float32)
    pred = student.predict(obs[None, :])[0]
    return pred.astype(np.float32)


def compute_weight(obs: np.ndarray, rho_indices: list[int], eps: float = 1e-3) -> float:
    if not rho_indices:
        return 1.0
    rho = obs[rho_indices]
    margin = 1.0 - rho
    min_margin = np.min(margin)
    return float(1.0 / (min_margin + eps))


def rollout(env: gym.Env, policy_student: DecisionTreeRegressor | None, expert: Agent, rho_indices: list[int], episodes: int, action_dim: int, use_expert_first: bool) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    observations = []
    expert_actions = []
    weights = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            acting_with_expert = use_expert_first and policy_student is None
            if acting_with_expert:
                act = expert_action(expert, obs)
            else:
                act = student_action(policy_student, obs, action_dim)
            next_obs, _, terminated, truncated, info = env.step(act)
            observations.append(obs.copy())
            expert_act = expert_action(expert, obs)
            expert_actions.append(expert_act)
            weights.append(compute_weight(obs, rho_indices))

            obs = next_obs
            done = terminated or truncated
            steps += 1
    return observations, expert_actions, weights


def evaluate_student(env: gym.Env, student: DecisionTreeRegressor, episodes: int = 1) -> float:
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            act = student_action(student, obs, env.action_space.shape[0])
            obs, reward, terminated, truncated, _ = env.step(act)
            total += float(reward)
            done = terminated or truncated
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint/final_PPO_bus14_R_0_0__I__1766145995_42074.tar")
    parser.add_argument("--env-id", type=str, default="bus14")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--episodes-per-iter", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Epsilon for criticalness weight")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env, env_args = make_env(args.env_id, seed=args.seed)
    expert, train_args = load_expert(args.checkpoint)
    action_dim = env.action_space.shape[0]

    # Feature indices for rho
    g2op_env = env.init_env
    feat_idx = get_feature_indices(g2op_env, train_args)
    rho_indices = feat_idx.get("rho", [])

    rho_indices = feat_idx.get("rho", [])
    print(f"DEBUG: Found {len(rho_indices)} rho indices starting at {rho_indices[0] if rho_indices else 'None'}.")

    student: DecisionTreeRegressor | None = None
    all_obs: list[np.ndarray] = []
    all_act: list[np.ndarray] = []
    all_w: list[float] = []

    for it in range(args.iterations):
        use_expert_first = it == 0
        obs_batch, act_batch, w_batch = rollout(env, student, expert, rho_indices, args.episodes_per_iter, action_dim, use_expert_first)
        all_obs.extend(obs_batch)
        all_act.extend(act_batch)
        all_w.extend(w_batch)
        print(f"Action stats: Mean={np.mean(np.abs(all_act)):.4f}, Max={np.max(np.abs(all_act)):.4f}")
        X = np.stack(all_obs)
        y = np.stack(all_act)
        sample_weight = np.array(all_w, dtype=np.float32)
        student = DecisionTreeRegressor(max_depth=args.max_depth, random_state=args.seed)
        student.fit(X, y, sample_weight=sample_weight)
        print(f"Iteration {it+1}: dataset size={len(all_obs)}, fitted tree depth={student.get_depth()}, leaves={student.get_n_leaves()}")

    total_reward = evaluate_student(env, student, episodes=1)
    print(f"Final student total reward (1 episode): {total_reward:.2f}")


if __name__ == "__main__":
    main()
