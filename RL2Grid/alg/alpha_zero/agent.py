from common.imports import *
from common.utils import Linear, th_act_fns


class AlphaZeroNetwork(nn.Module):
    """Shared policy-value network used by AlphaZero."""

    def __init__(self, obs_space: gym.Space, action_space: gym.Space,
                 hidden_dim: int = 512, num_hidden_layers: int = 3) -> None:
        super().__init__()
        self.obs_dim = int(np.prod(obs_space.shape))
        self.action_dim = int(np.prod(action_space.n))

        layers: List[nn.Module] = []
        in_features = self.obs_dim
        for _ in range(num_hidden_layers):
            layers.append(Linear(in_features, hidden_dim, 'relu'))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.policy_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim, 'relu'),
            nn.ReLU(),
            Linear(hidden_dim, self.action_dim)
        )
        self.value_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim, 'relu'),
            nn.ReLU(),
            Linear(hidden_dim, 1)
        )

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        flat = obs.view(obs.size(0), self.obs_dim)
        features = self.backbone(flat)
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value

    def get_eval_action(self, obs: th.Tensor) -> th.Tensor:
        logits, _ = self.forward(obs)
        return logits.argmax(dim=-1)

    def predict(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward(obs)
