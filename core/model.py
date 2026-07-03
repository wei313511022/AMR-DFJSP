from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy layer used by Rainbow."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = float(std_init)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.noise_enabled = True

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features).to(self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features).to(self.weight_mu.device)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_enabled:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class _ClassicQBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Optional[list[int]] = None):
        super().__init__()
        dims = [int(input_dim)] + [
            int(x)
            for x in (
                hidden_dims
                if hidden_dims is not None
                else [32, 64, 128, 256, 512, 1024, 512, 256, 128, 64, 32, 16, 8, 4]
            )
        ] + [1]
        self.hidden_dims = dims[1:-1]

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _RainbowQBackbone(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        noisy_std: float,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.num_atoms = int(num_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.uses_noisy_layers = True

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.value_hidden = NoisyLinear(128, 128, std_init=noisy_std)
        self.value_out = NoisyLinear(128, self.num_atoms, std_init=noisy_std)

        self.adv_hidden = NoisyLinear(128 + 64, 128, std_init=noisy_std)
        self.adv_out = NoisyLinear(128, self.num_atoms, std_init=noisy_std)

        self.register_buffer(
            "support",
            torch.linspace(self.v_min, self.v_max, self.num_atoms, dtype=torch.float32),
        )

    def reset_noise(self) -> None:
        for mod in self.modules():
            if isinstance(mod, NoisyLinear):
                mod.reset_noise()

    def set_noise_enabled(self, enabled: bool) -> None:
        for mod in self.modules():
            if isinstance(mod, NoisyLinear):
                mod.noise_enabled = bool(enabled)

    def _value_logits(self, state: torch.Tensor) -> torch.Tensor:
        state_feat = self.state_encoder(state)
        value_hidden = F.relu(self.value_hidden(state_feat))
        return self.value_out(value_hidden)

    def _advantage_logits(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(action)
        joined = torch.cat([state_feat, action_feat], dim=-1)
        adv_hidden = F.relu(self.adv_hidden(joined))
        return self.adv_out(adv_hidden)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        state = x[..., : self.state_dim]
        action = x[..., self.state_dim :]
        value_logits = self._value_logits(state)
        advantage_logits = self._advantage_logits(state, action)
        return value_logits + advantage_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        probs = F.softmax(logits, dim=-1)
        q = torch.sum(probs * self.support, dim=-1, keepdim=True)
        return q

    def action_logits(self, state: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action_feats.ndim == 1:
            action_feats = action_feats.unsqueeze(0)
        if state.shape[0] != 1:
            raise ValueError("action_logits expects one state and many action features.")

        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(action_feats)
        state_rep = state_feat.expand(action_feat.shape[0], -1)
        adv_hidden = F.relu(self.adv_hidden(torch.cat([state_rep, action_feat], dim=-1)))
        advantage_logits = self.adv_out(adv_hidden)

        value_hidden = F.relu(self.value_hidden(state_feat))
        value_logits = self.value_out(value_hidden).expand_as(advantage_logits)
        return value_logits + advantage_logits - advantage_logits.mean(dim=0, keepdim=True)

    def action_values(self, state: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        logits = self.action_logits(state, action_feats)
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.support, dim=-1)


class QNetwork(nn.Module):
    """
    Wrapper that can operate in classic DDQN mode or Rainbow mode.

    The default is Rainbow. Old checkpoints are still supported because
    load_state_dict infers whether the incoming state dict belongs to the
    classic MLP or the Rainbow backbone.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: Optional[int] = None,
        action_dim: int = 4,
        use_rainbow: bool = True,
        num_atoms: int = 51,
        v_min: float = -10000.0,
        v_max: float = 0.0,
        noisy_std: float = 0.5,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim if state_dim is not None else self.input_dim - self.action_dim)
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")

        self.num_atoms = int(num_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.noisy_std = float(noisy_std)
        self.classic_hidden_dims = [32, 64, 128, 256, 512, 1024, 512, 256, 128, 64, 32, 16, 8, 4]

        self.model_kind = ""
        self.net: nn.Module
        self._set_model_kind("rainbow" if use_rainbow else "classic")

    def _build_backbone(self, kind: str) -> nn.Module:
        if kind == "classic":
            return _ClassicQBackbone(self.input_dim, hidden_dims=self.classic_hidden_dims)
        if kind == "rainbow":
            return _RainbowQBackbone(
                self.state_dim,
                self.action_dim,
                self.num_atoms,
                self.v_min,
                self.v_max,
                self.noisy_std,
            )
        raise ValueError(f"Unknown model kind: {kind}")

    def _set_model_kind(self, kind: str, classic_hidden_dims: Optional[list[int]] = None) -> None:
        current_device = None
        current_dtype = None
        if hasattr(self, "net"):
            first_param = next(self.net.parameters(), None)
            if first_param is not None:
                current_device = first_param.device
                current_dtype = first_param.dtype

        if classic_hidden_dims is not None:
            self.classic_hidden_dims = [int(x) for x in classic_hidden_dims]
        self.model_kind = kind
        self.net = self._build_backbone(kind)

        if current_device is not None:
            self.net.to(device=current_device, dtype=current_dtype)

    @property
    def is_rainbow(self) -> bool:
        return self.model_kind == "rainbow"

    @property
    def uses_noisy_layers(self) -> bool:
        return self.is_rainbow and getattr(self.net, "uses_noisy_layers", False)

    def reset_noise(self) -> None:
        if hasattr(self.net, "reset_noise"):
            self.net.reset_noise()

    def set_noise_enabled(self, enabled: bool) -> None:
        if hasattr(self.net, "set_noise_enabled"):
            self.net.set_noise_enabled(enabled)

    def rainbow_config(self) -> dict:
        return {
            "model_kind": self.model_kind,
            "input_dim": self.input_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_atoms": self.num_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "noisy_std": self.noisy_std,
            "classic_hidden_dims": list(self.classic_hidden_dims),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _concat_state_actions(self, state: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action_feats.ndim == 1:
            action_feats = action_feats.unsqueeze(0)
        if state.shape[0] != 1:
            raise ValueError("Expected a single state when evaluating an action set.")
        state_rep = state.expand(action_feats.shape[0], -1)
        return torch.cat([state_rep, action_feats], dim=-1)

    def action_values(self, state: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        if self.is_rainbow:
            return self.net.action_values(state, action_feats)
        sa = self._concat_state_actions(state, action_feats)
        return self.net(sa).squeeze(-1)

    def action_logits(self, state: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        if not self.is_rainbow:
            raise RuntimeError("action_logits is only available in Rainbow mode.")
        return self.net.action_logits(state, action_feats)

    def chosen_action_value(
        self, state: torch.Tensor, action_feats: torch.Tensor, action_index: int
    ) -> torch.Tensor:
        values = self.action_values(state, action_feats)
        return values[int(action_index)]

    def chosen_action_logits(
        self, state: torch.Tensor, action_feats: torch.Tensor, action_index: int
    ) -> torch.Tensor:
        logits = self.action_logits(state, action_feats)
        return logits[int(action_index)]

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.net.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # type: ignore[override]
        inferred_kind = self._infer_model_kind(state_dict)
        if inferred_kind == "classic":
            inferred_hidden = self._infer_classic_hidden_dims(state_dict)
            rainbow_cfg = None
        else:
            inferred_hidden = None
            rainbow_cfg = self._infer_rainbow_config(state_dict)
            if rainbow_cfg is not None:
                self.num_atoms = int(rainbow_cfg["num_atoms"])
                self.v_min = float(rainbow_cfg["v_min"])
                self.v_max = float(rainbow_cfg["v_max"])
        if inferred_kind != self.model_kind or (
            inferred_hidden is not None and inferred_hidden != self.classic_hidden_dims
        ) or rainbow_cfg is not None:
            self._set_model_kind(inferred_kind, classic_hidden_dims=inferred_hidden)
        return self.net.load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _infer_model_kind(state_dict: dict) -> str:
        if any(k.startswith("net.") for k in state_dict.keys()):
            return "classic"
        if "support" in state_dict or any(k.startswith("state_encoder.") for k in state_dict.keys()):
            return "rainbow"
        # Fallback for checkpoints created through the wrapper itself.
        if any(k.startswith("value_hidden.") for k in state_dict.keys()):
            return "rainbow"
        return "classic"

    @staticmethod
    def _infer_classic_hidden_dims(state_dict: dict) -> list[int]:
        linears = []
        for key, tensor in state_dict.items():
            if not key.endswith(".weight"):
                continue
            if not key.startswith("net."):
                continue
            try:
                layer_idx = int(key.split(".")[1])
            except (IndexError, ValueError):
                continue
            if tensor.ndim != 2:
                continue
            linears.append((layer_idx, tuple(int(x) for x in tensor.shape)))

        if not linears:
            return [32, 64, 128, 256, 512, 1024, 512, 256, 128, 64, 32, 16, 8, 4]

        linears.sort(key=lambda item: item[0])
        return [shape[0] for _, shape in linears[:-1]]

    @staticmethod
    def _infer_rainbow_config(state_dict: dict) -> Optional[dict]:
        support = state_dict.get("support")
        if support is None:
            return None
        if getattr(support, "ndim", None) != 1:
            return None
        if int(support.shape[0]) <= 0:
            return None
        return {
            "num_atoms": int(support.shape[0]),
            "v_min": float(support[0].item()),
            "v_max": float(support[-1].item()),
        }
