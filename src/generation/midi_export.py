from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

from src.preprocessing.tokenizer import SimplePitchTokenizer


def piano_roll_to_pretty_midi(
	piano_roll: np.ndarray,
	fs: int = 16,
	program: int = 0,
	velocity: int = 90,
) -> pretty_midi.PrettyMIDI:
	if piano_roll.ndim != 2 or piano_roll.shape[1] != 128:
		raise ValueError("Expected piano_roll shape: [time, 128]")

	pm = pretty_midi.PrettyMIDI()
	instrument = pretty_midi.Instrument(program=program)
	min_dur = 1.0 / fs

	for pitch in range(128):
		active = np.where(piano_roll[:, pitch] > 0)[0]
		if active.size == 0:
			continue

		runs = np.split(active, np.where(np.diff(active) != 1)[0] + 1)
		for run in runs:
			start = float(run[0]) / fs
			end = float(run[-1] + 1) / fs
			if end - start < min_dur:
				end = start + min_dur
			instrument.notes.append(
				pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
			)

	pm.instruments.append(instrument)
	return pm


def tokens_to_pretty_midi(tokens: np.ndarray, tokenizer: SimplePitchTokenizer, fs: int = 16) -> pretty_midi.PrettyMIDI:
	clean = [t for t in tokens.tolist() if 0 <= int(t) <= 127 or int(t) == tokenizer.rest_token]
	roll = tokenizer.tokens_to_piano_roll(np.asarray(clean, dtype=np.int64))
	return piano_roll_to_pretty_midi(roll, fs=fs)


def save_midi(pm: pretty_midi.PrettyMIDI, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	pm.write(str(output_path))

