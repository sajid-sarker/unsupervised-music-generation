from __future__ import annotations

import torch


def sample_standard_normal(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
	return torch.randn(batch_size, latent_dim, device=device)


def interpolate_latent(z1: torch.Tensor, z2: torch.Tensor, steps: int) -> torch.Tensor:
	if steps < 2:
		raise ValueError("steps must be >= 2")
	alphas = torch.linspace(0.0, 1.0, steps, device=z1.device).unsqueeze(1)
	return (1.0 - alphas) * z1.unsqueeze(0) + alphas * z2.unsqueeze(0)

