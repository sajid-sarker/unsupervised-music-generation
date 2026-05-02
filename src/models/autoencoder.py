from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
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
		self.to_latent = nn.Linear(hidden_dim, latent_dim)
		self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
		self.decoder = nn.LSTM(
			input_size=input_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			dropout=dropout if num_layers > 1 else 0.0,
			batch_first=True,
		)
		self.output_proj = nn.Linear(hidden_dim, input_dim)

	def encode(self, x: torch.Tensor) -> torch.Tensor:
		_, (h_n, _) = self.encoder(x)
		return self.to_latent(h_n[-1])

	def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
		batch_size = z.size(0)
		init_h = torch.tanh(self.latent_to_hidden(z)).view(self.num_layers, batch_size, self.hidden_dim)
		init_c = torch.zeros_like(init_h)
		decoder_in = torch.zeros(batch_size, seq_len, self.input_dim, device=z.device)
		decoded, _ = self.decoder(decoder_in, (init_h, init_c))
		return torch.sigmoid(self.output_proj(decoded))

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		z = self.encode(x)
		recon = self.decode(z, seq_len=x.size(1))
		return recon, z

