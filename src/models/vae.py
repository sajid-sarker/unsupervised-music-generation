from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MusicVAE(nn.Module):
	def __init__(
		self,
		input_dim: int = 128,
		hidden_dim: int = 256,
		latent_dim: int = 64,
		num_layers: int = 2,
		dropout: float = 0.2,
	) -> None:
		super().__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim
		self.num_layers = num_layers

		self.encoder = nn.LSTM(
			input_size=input_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			dropout=dropout if num_layers > 1 else 0.0,
			batch_first=True,
		)
		self.mu_head = nn.Linear(hidden_dim, latent_dim)
		self.logvar_head = nn.Linear(hidden_dim, latent_dim)

		self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
		self.decoder = nn.LSTM(
			input_size=input_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			dropout=dropout if num_layers > 1 else 0.0,
			batch_first=True,
		)
		self.output_proj = nn.Linear(hidden_dim, input_dim)

	def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		_, (h_n, _) = self.encoder(x)
		h = h_n[-1]
		mu = self.mu_head(h)
		logvar = self.logvar_head(h)
		return mu, logvar

	def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
		batch_size = z.size(0)
		init_h = torch.tanh(self.latent_to_hidden(z)).view(self.num_layers, batch_size, self.hidden_dim)
		init_c = torch.zeros_like(init_h)
		decoder_in = torch.zeros(batch_size, seq_len, self.input_dim, device=z.device)
		decoded, _ = self.decoder(decoder_in, (init_h, init_c))
		return torch.sigmoid(self.output_proj(decoded))

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon = self.decode(z, seq_len=x.size(1))
		return recon, mu, logvar


def vae_loss(
	recon_x: torch.Tensor,
	x: torch.Tensor,
	mu: torch.Tensor,
	logvar: torch.Tensor,
	beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	recon = F.binary_cross_entropy(recon_x, x)
	kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
	total = recon + beta * kl
	return total, recon, kl

