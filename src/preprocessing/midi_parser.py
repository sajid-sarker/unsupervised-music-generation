from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pretty_midi

from src.config import DataConfig, PathConfig
from src.preprocessing.piano_roll import binarize_piano_roll, normalize_piano_roll, segment_piano_roll


def list_midi_files(root: Path) -> List[Path]:
	return sorted([p for p in root.rglob("*.mid")] + [p for p in root.rglob("*.midi")])


def midi_to_piano_roll(midi_path: Path, fs: int) -> np.ndarray:
	midi = pretty_midi.PrettyMIDI(str(midi_path))
	roll = midi.get_piano_roll(fs=fs).T
	roll = normalize_piano_roll(roll / 127.0)
	return binarize_piano_roll(roll)


def build_piano_roll_windows(
	midi_dir: Path,
	data_cfg: DataConfig,
) -> np.ndarray:
	windows = []
	midi_files = list_midi_files(midi_dir)
	for midi_file in midi_files:
		try:
			roll = midi_to_piano_roll(midi_file, fs=data_cfg.fs)
			chunks = segment_piano_roll(
				roll,
				window_size=data_cfg.window_size,
				stride=data_cfg.stride,
				min_active_notes=data_cfg.min_active_notes,
			)
			if len(chunks):
				windows.append(chunks)
		except Exception:
			continue

	if not windows:
		return np.zeros((0, data_cfg.window_size, 128), dtype=np.float32)
	return np.concatenate(windows, axis=0).astype(np.float32)


def save_windows(windows: np.ndarray, out_path: Path) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(out_path, windows)


def main() -> None:
	parser = argparse.ArgumentParser(description="Build piano-roll training windows from MIDI files.")
	parser.add_argument("--midi-dir", type=str, default=None)
	parser.add_argument("--output", type=str, default=None)
	parser.add_argument("--fs", type=int, default=16)
	parser.add_argument("--window-size", type=int, default=128)
	parser.add_argument("--stride", type=int, default=64)
	args = parser.parse_args()

	paths = PathConfig()
	data_cfg = DataConfig(fs=args.fs, window_size=args.window_size, stride=args.stride)
	midi_dir = Path(args.midi_dir) if args.midi_dir else paths.raw_midi_dir
	output = Path(args.output) if args.output else paths.processed_dir / "piano_roll_windows.npy"

	windows = build_piano_roll_windows(midi_dir, data_cfg)
	save_windows(windows, output)
	print(f"Saved {len(windows)} windows to {output}")


if __name__ == "__main__":
	main()

