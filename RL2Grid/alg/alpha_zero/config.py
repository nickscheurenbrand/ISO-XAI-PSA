from common.imports import *
from common.utils import str2bool


def get_alg_args() -> Namespace:
    """Parse AlphaZero-specific hyperparameters."""
    parser = ap.ArgumentParser()

    parser.add_argument("--az-total-iterations", type=int, default=256,
                        help="Number of AlphaZero training iterations.")
    parser.add_argument("--az-self-play-games", type=int, default=4,
                        help="Number of self-play games per iteration.")
    parser.add_argument("--az-max-episode-steps", type=int, default=288,
                        help="Maximum number of steps per self-play episode.")
    parser.add_argument("--az-mcts-simulations", type=int, default=32,
                        help="(Reserved) Number of MCTS simulations per move.")
    parser.add_argument("--az-temperature", type=float, default=1.0,
                        help="Temperature applied to policy logits before sampling actions.")
    parser.add_argument("--az-dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet alpha concentration for exploration noise.")
    parser.add_argument("--az-dirichlet-epsilon", type=float, default=0.25,
                        help="Mixing factor between policy and Dirichlet noise.")
    parser.add_argument("--az-replay-size", type=int, default=50000,
                        help="Size of the AlphaZero replay buffer.")
    parser.add_argument("--az-batch-size", type=int, default=256,
                        help="Batch size for policy/value updates.")
    parser.add_argument("--az-train-epochs", type=int, default=2,
                        help="Training epochs per iteration.")
    parser.add_argument("--az-learning-rate", type=float, default=1e-3,
                        help="Learning rate for AlphaZero network.")
    parser.add_argument("--az-weight-decay", type=float, default=1e-4,
                        help="Weight decay for AlphaZero optimizer.")
    parser.add_argument("--az-value-loss-weight", type=float, default=1.0,
                        help="Weight applied to the value loss term.")
    parser.add_argument("--az-policy-loss-weight", type=float, default=1.0,
                        help="Weight applied to the policy loss term.")
    parser.add_argument("--az-eval-freq", type=int, default=8,
                        help="Frequency (in iterations) of deterministic evaluations.")
    parser.add_argument("--az-gamma", type=float, default=0.99,
                        help="Discount factor for computing self-play returns.")
    parser.add_argument("--az-max-grad-norm", type=float, default=5.0,
                        help="Gradient clipping value for AlphaZero updates.")

    return parser.parse_known_args()[0]
