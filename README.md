# ISO-XAI-PSA

Training command used for the bus14 PPO checkpoint:
- `python main.py --alg PPO --env-id bus14 --action-type redispatch`

Quick commands to run the three main explainability scripts. Run them from the `RL2Grid/` directory (they expect checkpoints under `RL2Grid/checkpoint/`).

## SHAP summary
- Command: `python explain_shap.py --checkpoint final_PPO_bus14`
- What it does: loads the PPO checkpoint, gathers background/test rollouts, and writes SHAP summary plots (`shap_summary_*.png`).

## SHAP noise stability
- Command: `python shap_noise_stability.py --chekpoint final_PPO_bus14`
- What it does: tests SHAP robustness to tiny observation noise, printing per-sample stats and saving bar plots (`shap_noise_stability_mean*.png`).

## Counterfactual generation
- Command: `python run_counterfactual.py --checkpoint checkpoint/final_PPO_bus14.tar`
- What it does: trains the counterfactual modules on fresh rollouts from the PPO agent and writes comparison CSV/plots (`counterfactual_comparison.csv`, `counterfactual_diff.png`).