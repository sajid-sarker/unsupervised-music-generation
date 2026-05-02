from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def reconstruction_mse(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
	return F.mse_loss(pred, target)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
	return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def token_perplexity(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int | None = None) -> float:
	vocab_size = logits.shape[-1]
	flat_logits = logits.reshape(-1, vocab_size)
	flat_targets = targets.reshape(-1)
	ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index)
	return float(torch.exp(ce).item())


def save_loss_curve(values: list[float], output_path: Path, title: str, ylabel: str) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(8, 4))
	plt.plot(np.arange(1, len(values) + 1), values, marker="o", linewidth=1.5)
	plt.title(title)
	plt.xlabel("Epoch")
	plt.ylabel(ylabel)
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()

