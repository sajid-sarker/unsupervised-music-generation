from __future__ import annotations

import numpy as np


def onset_density(piano_roll: np.ndarray) -> float:
	if piano_roll.ndim != 2 or piano_roll.shape[1] != 128:
		raise ValueError("Expected piano_roll shape: [time, 128]")
	activity = (piano_roll.sum(axis=1) > 0).astype(np.float32)
	return float(activity.mean())


def rhythm_similarity(reference_roll: np.ndarray, generated_roll: np.ndarray) -> float:
	ref_density = onset_density(reference_roll)
	gen_density = onset_density(generated_roll)
	return max(0.0, 1.0 - abs(ref_density - gen_density))

