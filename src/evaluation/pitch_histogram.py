from __future__ import annotations

import numpy as np


def pitch_histogram(piano_roll: np.ndarray) -> np.ndarray:
	if piano_roll.ndim != 2 or piano_roll.shape[1] != 128:
		raise ValueError("Expected piano_roll shape: [time, 128]")
	hist = piano_roll.sum(axis=0).astype(np.float64)
	total = hist.sum()
	if total == 0:
		return np.zeros(128, dtype=np.float64)
	return hist / total


def histogram_l1_distance(a: np.ndarray, b: np.ndarray) -> float:
	return float(np.abs(a - b).sum())


def compare_pitch_distributions(reference_roll: np.ndarray, generated_roll: np.ndarray) -> float:
	ref = pitch_histogram(reference_roll)
	gen = pitch_histogram(generated_roll)
	return histogram_l1_distance(ref, gen)

