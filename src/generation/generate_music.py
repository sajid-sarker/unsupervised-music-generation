from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from src.config import AEConfig, PathConfig, TransformerConfig, VAEConfig, get_device
from src.generation.midi_export import save_midi, piano_roll_to_pretty_midi, tokens_to_pretty_midi
from src.generation.sample_latent import sample_standard_normal
from src.models.autoencoder import LSTMAutoencoder
from src.models.transformer import CausalTransformer
from src.models.vae import MusicVAE
from src.preprocessing.tokenizer import SimplePitchTokenizer


def load_checkpoint(path: Path, device: torch.device) -> dict:
	if not path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {path}")
	return torch.load(path, map_location=device)


@torch.no_grad()
def generate_from_ae(ckpt_path: Path, out_dir: Path, num_samples: int, seq_len: int, fs: int) -> None:
	device = get_device()
	cfg = AEConfig()
	model = LSTMAutoencoder(
		input_dim=cfg.input_dim,
		hidden_dim=cfg.hidden_dim,
		latent_dim=cfg.latent_dim,
		num_layers=cfg.num_layers,
		dropout=cfg.dropout,
	).to(device)
	payload = load_checkpoint(ckpt_path, device)
	model.load_state_dict(payload["model_state_dict"])
	model.eval()

	z = sample_standard_normal(num_samples, cfg.latent_dim, device)
	rolls = model.decode(z, seq_len=seq_len).cpu().numpy()
	rolls = (rolls > 0.5).astype(np.float32)

	for i, roll in enumerate(rolls):
		pm = piano_roll_to_pretty_midi(roll, fs=fs)
		save_midi(pm, out_dir / f"ae_sample_{i+1:02d}.mid")


@torch.no_grad()
def generate_from_vae(ckpt_path: Path, out_dir: Path, num_samples: int, seq_len: int, fs: int) -> None:
	device = get_device()
	cfg = VAEConfig()
	model = MusicVAE(
		input_dim=cfg.input_dim,
		hidden_dim=cfg.hidden_dim,
		latent_dim=cfg.latent_dim,
		num_layers=cfg.num_layers,
		dropout=cfg.dropout,
	).to(device)
	payload = load_checkpoint(ckpt_path, device)
	model.load_state_dict(payload["model_state_dict"])
	model.eval()

	z = sample_standard_normal(num_samples, cfg.latent_dim, device)
	rolls = model.decode(z, seq_len=seq_len).cpu().numpy()
	rolls = (rolls > 0.5).astype(np.float32)

	for i, roll in enumerate(rolls):
		pm = piano_roll_to_pretty_midi(roll, fs=fs)
		save_midi(pm, out_dir / f"vae_sample_{i+1:02d}.mid")


@torch.no_grad()
def generate_from_transformer(
	ckpt_path: Path,
	out_dir: Path,
	num_samples: int,
	seq_len: int,
	temperature: float,
	fs: int,
) -> None:
	device = get_device()
	tcfg = TransformerConfig()
	tok = SimplePitchTokenizer()
	model = CausalTransformer(
		vocab_size=tcfg.vocab_size,
		d_model=tcfg.d_model,
		nhead=tcfg.nhead,
		num_layers=tcfg.num_layers,
		dim_feedforward=tcfg.dim_feedforward,
		dropout=tcfg.dropout,
		max_seq_len=tcfg.max_seq_len,
	).to(device)

	payload = load_checkpoint(ckpt_path, device)
	model.load_state_dict(payload["model_state_dict"])
	model.eval()

	start = torch.full((num_samples, 1), tok.bos_token, dtype=torch.long, device=device)
	generated = model.generate(start, max_new_tokens=seq_len, temperature=temperature)
	generated = generated[:, 1:].cpu().numpy()

	for i, seq in enumerate(generated):
		pm = tokens_to_pretty_midi(seq, tokenizer=tok, fs=fs)
		save_midi(pm, out_dir / f"transformer_sample_{i+1:02d}.mid")


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate MIDI samples from trained models.")
	parser.add_argument("--model", choices=["ae", "vae", "transformer"], required=True)
	parser.add_argument("--checkpoint", type=str, default=None)
	parser.add_argument("--num-samples", type=int, default=5)
	parser.add_argument("--seq-len", type=int, default=128)
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--fs", type=int, default=16)
	args = parser.parse_args()

	paths = PathConfig()
	out_dir = paths.generated_midi_dir
	out_dir.mkdir(parents=True, exist_ok=True)

	if args.model == "ae":
		ckpt = Path(args.checkpoint) if args.checkpoint else paths.checkpoints_dir / "ae.pt"
		generate_from_ae(ckpt, out_dir, args.num_samples, args.seq_len, args.fs)
	elif args.model == "vae":
		ckpt = Path(args.checkpoint) if args.checkpoint else paths.checkpoints_dir / "vae.pt"
		generate_from_vae(ckpt, out_dir, args.num_samples, args.seq_len, args.fs)
	else:
		ckpt = Path(args.checkpoint) if args.checkpoint else paths.checkpoints_dir / "transformer.pt"
		generate_from_transformer(ckpt, out_dir, args.num_samples, args.seq_len, args.temperature, args.fs)

	print(f"Generated {args.num_samples} samples with {args.model} in {out_dir}")


if __name__ == "__main__":
	main()

