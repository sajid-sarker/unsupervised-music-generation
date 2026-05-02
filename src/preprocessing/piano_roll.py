from __future__ import annotations

from typing import List

import numpy as np


def normalize_piano_roll(piano_roll: np.ndarray) -> np.ndarray:
	if piano_roll.size == 0:
		return piano_roll.astype(np.float32)
	piano_roll = np.clip(piano_roll, 0.0, 1.0)
	return piano_roll.astype(np.float32)


def binarize_piano_roll(piano_roll: np.ndarray, threshold: float = 0.0) -> np.ndarray:
	return (piano_roll > threshold).astype(np.float32)


def segment_piano_roll(
	piano_roll: np.ndarray,
	window_size: int,
	stride: int,
	min_active_notes: int = 1,
) -> np.ndarray:
	if piano_roll.ndim != 2 or piano_roll.shape[1] != 128:
		raise ValueError("Expected piano_roll shape: [time, 128]")

	windows: List[np.ndarray] = []
	total_steps = piano_roll.shape[0]

	if total_steps < window_size:
		pad_len = window_size - total_steps
		padded = np.pad(piano_roll, ((0, pad_len), (0, 0)), mode="constant")
		if padded.sum() >= min_active_notes:
			windows.append(padded)
		return np.asarray(windows, dtype=np.float32)

	for start in range(0, total_steps - window_size + 1, stride):
		window = piano_roll[start : start + window_size]
		if window.sum() >= min_active_notes:
			windows.append(window)

	if not windows:
		return np.zeros((0, window_size, 128), dtype=np.float32)

	return np.asarray(windows, dtype=np.float32)

