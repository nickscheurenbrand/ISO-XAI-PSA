"""Counterfactual state generation for PPO agents.

This module implements the Counterfactual State technique described by Olson et al. (2019),
adapted for frozen PPO agents trained on redispatch (continuous control) tasks.

Observations are assumed to be vector-valued (e.g., concatenated grid measurements) rather
than images. The auxiliary networks operate on flattened tensors that share the same shape
as the PPO observation space. The PPO agent remains frozen while the counterfactual encoder,
generator, discriminator, and Wasserstein auto-encoder are trained.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _agent_feature_extractor(agent: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    if hasattr(agent, "feature_extractor"):
        return agent.feature_extractor
    if hasattr(agent, "actor") and hasattr(agent.actor, "features"):
        return agent.actor.features
    def identity(x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1) if x.dim() > 2 else x
    return identity


def _agent_policy_head(agent: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    if hasattr(agent, "policy_head"):
        return agent.policy_head
    if hasattr(agent, "actor"):
        return agent.actor
    raise AttributeError("Agent must expose a policy head (policy_head or actor)")


def _flatten(states: torch.Tensor) -> torch.Tensor:
    if states.dim() <= 2:
        return states
    return states.view(states.size(0), -1)


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    return -(probs * log_probs).sum(dim=-1).mean()


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    def _rbf(a, b):
        a2 = (a ** 2).sum(dim=1, keepdim=True)
        b2 = (b ** 2).sum(dim=1, keepdim=True)
        dist = a2 + b2.T - 2 * a @ b.T
        return torch.exp(-dist / (2 * sigma ** 2))

    k_xx = _rbf(x, x)
    k_yy = _rbf(y, y)
    k_xy = _rbf(x, y)
    m = x.size(0)
    n = y.size(0)
    mmd = (k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1) + 1e-8)
    mmd += (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1) + 1e-8)
    mmd -= 2 * k_xy.mean()
    return mmd


def _confidence_loss(policy_logits: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
    if target_action.ndim == 1:
        target_action = target_action.unsqueeze(0)
    target_action = target_action.to(policy_logits.device)
    if target_action.shape == policy_logits.shape:
        return F.mse_loss(policy_logits, target_action)
    probs = torch.softmax(policy_logits, dim=-1)
    loss = -torch.log((probs * target_action).sum(dim=-1) + 1e-8)
    return loss.mean()


# -----------------------------------------------------------------------------
# Neural network blocks
# -----------------------------------------------------------------------------


class Encoder(nn.Module):
    """Encodes vector observations into action-invariant embeddings."""

    def __init__(self, obs_dim: int, encoding_dim: int, hidden_dims: Tuple[int, ...] = (512, 512)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_features = obs_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden
        layers.append(nn.Linear(in_features, encoding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class Generator(nn.Module):
    """Decodes encodings + policy signals back into vector observations."""

    def __init__(self, encoding_dim: int, policy_dim: int, obs_dim: int, hidden_dims: Tuple[int, ...] = (512, 512)) -> None:
        super().__init__()
        latent_dim = encoding_dim + policy_dim
        layers: List[nn.Module] = []
        in_features = latent_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden
        layers.append(nn.Linear(in_features, obs_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, encoding: torch.Tensor, policy: torch.Tensor) -> torch.Tensor:
        latent = torch.cat([encoding, policy], dim=1)
        return self.net(latent)


class Discriminator(nn.Module):
    """Predicts policy logits from the encoder output."""

    def __init__(self, encoding_dim: int, policy_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, policy_dim),
        )

    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        return self.net(encoding)


class WassersteinAE(nn.Module):
    """Projects PPO latents onto a compact manifold via a WAE."""

    def __init__(self, latent_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(projection_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return self.encoder(z)

    def decode(self, z_w: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_w)


# -----------------------------------------------------------------------------
# Counterfactual trainer
# -----------------------------------------------------------------------------


@dataclass
class CounterfactualTrainer:
    """Trains counterfactual modules against a frozen PPO agent."""

    agent: nn.Module
    dataloader: DataLoader
    encoder: Encoder
    generator: Generator
    discriminator: Discriminator
    wae: WassersteinAE
    device: torch.device = _default_device()
    lambda_adv: float = 1.0
    lambda_mmd: float = 10.0

    def __post_init__(self) -> None:
        self.agent.eval()
        for p in self.agent.parameters():
            p.requires_grad_(False)

        self._feat_fn = _agent_feature_extractor(self.agent)
        self._policy_fn = _agent_policy_head(self.agent)

        self.encoder.to(self.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.wae.to(self.device)

        self.opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.opt_generator = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.opt_wae = torch.optim.Adam(self.wae.parameters(), lr=1e-4)

    def _feature_extractor(self, states: torch.Tensor) -> torch.Tensor:
        return self._feat_fn(states)

    def _policy_head(self, latents: torch.Tensor) -> torch.Tensor:
        return self._policy_fn(latents)

    def train_epoch(self) -> Dict[str, float]:
        metrics = {"loss_ae": 0.0, "loss_disc": 0.0, "loss_adv": 0.0, "loss_wae": 0.0}
        batches = 0

        for states in self.dataloader:
            if isinstance(states, (tuple, list)):
                states = states[0]
            states = states.to(self.device)
            flat_states = _flatten(states)
            batches += 1

            with torch.no_grad():
                z = self._feature_extractor(states)
                policy_logits = self._policy_head(z)

            encoding = self.encoder(flat_states)
            recon = self.generator(encoding, policy_logits.detach())
            loss_ae = F.mse_loss(recon, flat_states)

            disc_pred = self.discriminator(encoding.detach())
            loss_disc = F.mse_loss(disc_pred, policy_logits.detach())

            self.opt_discriminator.zero_grad()
            loss_disc.backward()
            self.opt_discriminator.step()

            adv_logits = self.discriminator(encoding)
            loss_adv = -_entropy_from_logits(adv_logits)

            self.opt_encoder.zero_grad()
            self.opt_generator.zero_grad()
            total_gen_loss = loss_ae + self.lambda_adv * loss_adv
            total_gen_loss.backward()
            self.opt_encoder.step()
            self.opt_generator.step()

            z_w = self.wae.encode(z.detach())
            z_recon = self.wae.decode(z_w)
            prior = torch.randn_like(z_w)
            loss_wae = F.mse_loss(z_recon, z.detach()) + self.lambda_mmd * _mmd_rbf(z_w, prior)

            self.opt_wae.zero_grad()
            loss_wae.backward()
            self.opt_wae.step()

            metrics["loss_ae"] += loss_ae.item()
            metrics["loss_disc"] += loss_disc.item()
            metrics["loss_adv"] += loss_adv.item()
            metrics["loss_wae"] += loss_wae.item()

        for key in metrics:
            metrics[key] /= max(1, batches)
        return metrics


# -----------------------------------------------------------------------------
# Counterfactual generation
# -----------------------------------------------------------------------------


def generate_counterfactual(
    state: torch.Tensor,
    target_action: torch.Tensor,
    agent: nn.Module,
    encoder: Encoder,
    generator: Generator,
    wae: WassersteinAE,
    lr: float = 5e-2,
    steps: int = 250,
    recon_weight: float = 1.0,
    action_weight: float = 5.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate a counterfactual state that makes the agent prefer `target_action`."""

    device = device or _default_device()
    if state.dim() == 1:
        state = state.unsqueeze(0)
    state = state.to(device)
    original_shape = state.shape
    flat_state = _flatten(state)
    agent.eval()
    encoder.eval()
    generator.eval()
    wae.eval()

    feat_fn = _agent_feature_extractor(agent)
    policy_fn = _agent_policy_head(agent)

    with torch.no_grad():
        z = feat_fn(state)

    z = z.detach()
    z_w = wae.encode(z).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z_w], lr=lr)

    for _ in range(steps):
        decoded = wae.decode(z_w)
        recon_loss = F.mse_loss(decoded, z)
        policy_logits = policy_fn(decoded)
        act_loss = _confidence_loss(policy_logits, target_action.to(device))
        loss = recon_weight * recon_loss + action_weight * act_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        z_star = wae.decode(z_w)
        policy_star = policy_fn(z_star)
        enc_state = encoder(flat_state)
        cf_flat = generator(enc_state, policy_star)
    return cf_flat.view(original_shape)
