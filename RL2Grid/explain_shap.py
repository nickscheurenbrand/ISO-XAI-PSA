import argparse
import numpy as np
import torch as th
import shap
import matplotlib.pyplot as plt
import os
import sys
import gymnasium as gym

# Add script directory to path so imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from alg.ppo.agent import Agent
from env.utils import auxiliary_make_env, load_config
from common.utils import set_torch, set_random_seed

def get_feature_names(g2op_env, args, config):
    env_id = args.env_id
    env_config = config['environments']
    state_attrs = config['state_attrs']
    
    # Replicate logic from env/utils.py to get obs_attrs
    obs_attrs = list(state_attrs['default']) # Copy
    if env_config[env_id]['maintenance']: obs_attrs += state_attrs['maintenance']

    obs_attrs += state_attrs['redispatch']
    if env_config[env_id]['renewable'] : 
        obs_attrs += state_attrs['curtailment']
    if env_config[env_id]['battery']:
        obs_attrs += state_attrs['storage']
            
    feature_names = []
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
            # Fallback for unknown attributes, try to guess size or just append attr name
            # This might cause length mismatch if attr is a vector
            print(f"Warning: Unknown attribute {attr}, cannot determine feature names.")
            names = [attr]
            
        feature_names.extend(names)
        
    return feature_names

def main():
    parser = argparse.ArgumentParser(description="Compute SHAP values for a trained PPO agent.")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the run to load (without .tar)")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of background samples")
    parser.add_argument("--n-test", type=int, default=10, help="Number of test samples to explain")
    
    args, unknown = parser.parse_known_args()
    
    checkpoint_path = f"RL2Grid/checkpoint/{args.run_name}.tar"
    # Fallback to checking current dir if not found
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"checkpoint/{args.run_name}.tar"
        
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = th.load(checkpoint_path, map_location='cpu')
    
    # The checkpoint contains 'args' which is a Namespace or dict.
    train_args = checkpoint['args']
    
    # Override args for evaluation
    train_args.cuda = False
    train_args.n_envs = 1 # Force single env for explanation

    # Set up environment
    print("Creating environment...")
    # We need to ensure we are in the right directory for relative paths in env config
    # auxiliary_make_env uses args.env_config_path which defaults to "scenario.json"
    # It expects it in env/scenario.json usually or relative.
    # Let's assume the script is run from the root of the repo.
    
    def make_env():
        env, _ = auxiliary_make_env(train_args, resume_run=False, test=True)
        return env

    env = gym.vector.SyncVectorEnv([make_env])
    
    # Initialize Agent
    print("Initializing agent...")
    continuous_actions = True
    agent = Agent(env, train_args, continuous_actions)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    agent.eval()
    
    # Prepare background data
    print(f"Collecting {args.n_samples} background samples...")
    background_obs = []
    obs, _ = env.reset()
    
    # We need to flatten observation if it's not already flat, but Agent expects flat usually?
    # Agent.init uses np.prod(envs.single_observation_space.shape) for input dim.
    # But the forward pass expects the shape that comes out of env.
    # Let's check shape.
    print(f"Observation shape: {obs.shape}")
    
    for _ in range(args.n_samples):
        background_obs.append(obs[0])
        # Get action from agent to step env
        with th.no_grad():
            t_obs = th.tensor(obs).float()
            # Use get_eval_action for deterministic path if possible, or get_action
            action = agent.get_eval_continuous_action(t_obs)
                
        # Step environment
        # Action needs to be numpy
        action_np = action.detach().numpy()
        obs, _, terminated, truncated, _ = env.step(action_np)
            
    background_data = np.stack(background_obs)
    print(f"Background data shape: {background_data.shape}")

    # Reset the environment so test samples come from an independent rollout
    obs, _ = env.reset()
    
    # Prepare test data
    print(f"Collecting {args.n_test} test samples to explain...")
    test_obs = []
    for _ in range(args.n_test):
        test_obs.append(obs[0])
        with th.no_grad():
            t_obs = th.tensor(obs).float()
            action = agent.get_eval_continuous_action(t_obs)
        
        action_np = action.detach().numpy()
        obs, _, terminated, truncated, _ = env.step(action_np)
    
    test_data = np.stack(test_obs)

    background_tensor = th.tensor(background_data, dtype=th.float32)
    test_tensor = th.tensor(test_data, dtype=th.float32)

    # Initialize Explainer
    print("Computing SHAP values using DeepExplainer...")
    explainer = shap.DeepExplainer(agent.actor, background_tensor)
    shap_values_raw = explainer.shap_values(test_tensor, check_additivity=False)
    
    print("SHAP values computed.")
    shap_array = np.array(shap_values_raw)
    print(f"SHAP values array shape: {shap_array.shape}")

    if shap_array.ndim != 3:
        raise ValueError(
            f"Expected SHAP output with shape (n_samples, n_features, n_generators), got {shap_array.shape}"
        )

    n_samples, n_features = test_data.shape
    if shap_array.shape[0] != n_samples or shap_array.shape[1] != n_features:
        raise ValueError(
            f"SHAP output first two dimensions {shap_array.shape[:2]} do not match data {test_data.shape}"
        )

    shap_values_list = [shap_array[:, :, i] for i in range(shap_array.shape[2])]
    
    try:
        # Access the inner grid2op environment
        # env is SyncVectorEnv, env.envs[0] is the GymEnv
        # But SyncVectorEnv might not expose envs directly if it's a lambda
        # Actually SyncVectorEnv stores envs in self.envs
        gym_env = env.envs[0]
        g2op_env = gym_env.init_env
        
        config = load_config(train_args.env_config_path)
        feature_names = get_feature_names(g2op_env, train_args, config)
        
        if len(feature_names) != test_data.shape[1]:
            print(f"Warning: Feature names length ({len(feature_names)}) does not match data dimension ({test_data.shape[1]}). Using default names.")
            feature_names = None
    except Exception as e:
        print(f"Warning: Could not retrieve feature names: {e}")
        feature_names = None
        generator_labels = None

    # Save summary plots
    n_outputs = len(shap_values_list)
    print(f"Model has {n_outputs} generator outputs. SHAP values normalized as list of matrices.")

    # Calculate global importance (sum of mean abs SHAP across all outputs)
    mean_shap = np.sum([np.abs(sv).mean(axis=0) for sv in shap_values_list], axis=0)

    # 1. Global Feature Importance (Bar Chart)
    output_bar = "shap_summary_bar.pdf"
    print(f"Saving global feature importance (all actions) to {output_bar}...")
    plt.figure(figsize=(12, 9))
    generator_titles = [f"Generator {i+1}" for i in range(n_outputs)]
    shap.summary_plot(
        shap_values_list,
        test_data,
        feature_names=feature_names,
        plot_type="bar",
        class_names=generator_titles,
        max_display=10,
        show=False,
    )
    plt.gca().set_title("")
    
    #plt.suptitle("Global importance (mean |SHAP| across generators)", fontsize=14, y=0.98)
    plt.xlabel("mean(|SHAP value|)")
    plt.tight_layout()
    
    # Save the figure as pdf
    plt.savefig(output_bar, bbox_inches='tight', format='pdf')
    plt.close()

    # 2. Beeswarm per generator
    print("Saving beeswarm plots for each generator...")
    for idx, shap_vals in enumerate(shap_values_list):
        label = generator_titles[idx] if idx < len(generator_titles) else f"Generator {idx+1}"
        output_bee = f"shap_summary_beeswarm_gen{idx+1}.pdf"
        print(f" - {label} -> {output_bee}")
        plt.figure()
        shap.summary_plot(shap_vals, test_data, feature_names=feature_names, max_display=10, show=False)
        #plt.title(label)
        plt.savefig(output_bee, bbox_inches='tight', format='pdf')
        plt.close()
        
    print("Done.")

if __name__ == "__main__":
    main()
