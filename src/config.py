from dataclasses import dataclass
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PathConfig:
	raw_midi_dir: Path = PROJECT_ROOT / "data" / "raw_midi"
	processed_dir: Path = PROJECT_ROOT / "data" / "processed"
	split_dir: Path = PROJECT_ROOT / "data" / "train_test_split"
	outputs_dir: Path = PROJECT_ROOT / "outputs"
	plots_dir: Path = PROJECT_ROOT / "outputs" / "plots"
	generated_midi_dir: Path = PROJECT_ROOT / "outputs" / "generated_midis"
	checkpoints_dir: Path = PROJECT_ROOT / "outputs" / "checkpoints"


@dataclass(frozen=True)
class DataConfig:
	fs: int = 16
	window_size: int = 128
	stride: int = 64
	min_active_notes: int = 4


@dataclass(frozen=True)
class AEConfig:
	input_dim: int = 128
	hidden_dim: int = 256
	latent_dim: int = 64
	num_layers: int = 2
	dropout: float = 0.2
	lr: float = 1e-3
	batch_size: int = 32
	epochs: int = 20


@dataclass(frozen=True)
class VAEConfig:
	input_dim: int = 128
	hidden_dim: int = 256
	latent_dim: int = 64
	num_layers: int = 2
	dropout: float = 0.2
	beta: float = 0.1
	lr: float = 1e-3
	batch_size: int = 32
	epochs: int = 25


@dataclass(frozen=True)
class TransformerConfig:
	vocab_size: int = 132
	d_model: int = 256
	nhead: int = 8
	num_layers: int = 4
	dim_feedforward: int = 512
	dropout: float = 0.1
	max_seq_len: int = 512
	lr: float = 3e-4
	batch_size: int = 32
	epochs: int = 20


def get_device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dirs(paths: PathConfig) -> None:
	paths.processed_dir.mkdir(parents=True, exist_ok=True)
	paths.outputs_dir.mkdir(parents=True, exist_ok=True)
	paths.plots_dir.mkdir(parents=True, exist_ok=True)
	paths.generated_midi_dir.mkdir(parents=True, exist_ok=True)
	paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

