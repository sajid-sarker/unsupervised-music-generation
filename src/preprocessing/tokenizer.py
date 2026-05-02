from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimplePitchTokenizer:
	pad_token: int = 128
	bos_token: int = 129
	eos_token: int = 130
	rest_token: int = 131
	vocab_size: int = 132

	def piano_roll_to_tokens(self, piano_roll: np.ndarray) -> np.ndarray:
		if piano_roll.ndim != 2 or piano_roll.shape[1] != 128:
			raise ValueError("Expected piano_roll shape: [time, 128]")

		tokens = []
		for frame in piano_roll:
			active = np.where(frame > 0)[0]
			if active.size == 0:
				tokens.append(self.rest_token)
			else:
				tokens.append(int(active.max()))
		return np.asarray(tokens, dtype=np.int64)

	def tokens_to_piano_roll(self, tokens: np.ndarray) -> np.ndarray:
		roll = np.zeros((len(tokens), 128), dtype=np.float32)
		for t, token in enumerate(tokens):
			if 0 <= int(token) <= 127:
				roll[t, int(token)] = 1.0
		return roll

	def add_special_tokens(self, tokens: np.ndarray) -> np.ndarray:
		return np.concatenate(
			[np.array([self.bos_token], dtype=np.int64), tokens, np.array([self.eos_token], dtype=np.int64)]
		)


def segment_token_sequence(tokens: np.ndarray, window_size: int, stride: int) -> np.ndarray:
	if len(tokens) < window_size:
		padded = np.pad(tokens, (0, window_size - len(tokens)), constant_values=128)
		return padded[None, :].astype(np.int64)

	windows = []
	for start in range(0, len(tokens) - window_size + 1, stride):
		windows.append(tokens[start : start + window_size])
	return np.asarray(windows, dtype=np.int64)

