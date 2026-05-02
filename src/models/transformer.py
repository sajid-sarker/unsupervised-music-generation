from __future__ import annotations

import torch
from torch import nn


class CausalTransformer(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		d_model: int = 256,
		nhead: int = 8,
		num_layers: int = 4,
		dim_feedforward: int = 512,
		dropout: float = 0.1,
		max_seq_len: int = 512,
	) -> None:
		super().__init__()
		self.vocab_size = vocab_size
		self.max_seq_len = max_seq_len

		self.token_emb = nn.Embedding(vocab_size, d_model)
		self.pos_emb = nn.Embedding(max_seq_len, d_model)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.norm = nn.LayerNorm(d_model)
		self.head = nn.Linear(d_model, vocab_size)

	def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
		return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

	def forward(self, tokens: torch.Tensor) -> torch.Tensor:
		bsz, seq_len = tokens.shape
		if seq_len > self.max_seq_len:
			raise ValueError(f"Input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")

		pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(bsz, -1)
		x = self.token_emb(tokens) + self.pos_emb(pos)
		mask = self._causal_mask(seq_len, tokens.device)
		x = self.encoder(x, mask=mask)
		x = self.norm(x)
		return self.head(x)

	@torch.no_grad()
	def generate(
		self,
		start_tokens: torch.Tensor,
		max_new_tokens: int,
		temperature: float = 1.0,
		top_k: int | None = 20,
	) -> torch.Tensor:
		self.eval()
		generated = start_tokens

		for _ in range(max_new_tokens):
			context = generated[:, -self.max_seq_len :]
			logits = self.forward(context)[:, -1, :]
			logits = logits / max(temperature, 1e-6)

			if top_k is not None:
				values, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]))
				logits[logits < values[:, [-1]]] = -float("inf")

			probs = torch.softmax(logits, dim=-1)
			next_token = torch.multinomial(probs, num_samples=1)
			generated = torch.cat([generated, next_token], dim=1)

		return generated

