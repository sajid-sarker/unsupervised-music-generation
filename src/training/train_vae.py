from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from src.config import DataConfig, PathConfig, VAEConfig, ensure_output_dirs, get_device
from src.evaluation.metrics import save_loss_curve
from src.models.vae import MusicVAE, vae_loss
from src.preprocessing.midi_parser import build_piano_roll_windows, save_windows


def load_or_build_dataset(paths: PathConfig, data_cfg: DataConfig) -> np.ndarray:
	dataset_path = paths.processed_dir / "piano_roll_windows.npy"
	if dataset_path.exists():
		return np.load(dataset_path)

	windows = build_piano_roll_windows(paths.raw_midi_dir, data_cfg)
	save_windows(windows, dataset_path)
	return windows


def train(args: argparse.Namespace) -> None:
	paths = PathConfig()
	data_cfg = DataConfig(window_size=args.window_size, stride=args.stride)
	cfg = VAEConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, beta=args.beta)
	ensure_output_dirs(paths)

	windows = load_or_build_dataset(paths, data_cfg)
	if len(windows) == 0:
		raise RuntimeError("No training windows found. Add MIDI files to data/raw_midi first.")

	tensor = torch.tensor(windows, dtype=torch.float32)
	dataset = TensorDataset(tensor)
	train_size = int(0.9 * len(dataset))
	val_size = len(dataset) - train_size
	train_set, val_set = random_split(dataset, [train_size, val_size])

	train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
	val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

	device = get_device()
	model = MusicVAE(
		input_dim=cfg.input_dim,
		hidden_dim=cfg.hidden_dim,
		latent_dim=cfg.latent_dim,
		num_layers=cfg.num_layers,
		dropout=cfg.dropout,
	).to(device)
	opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

	train_losses = []
	val_losses = []
	train_kls = []

	for epoch in range(cfg.epochs):
		model.train()
		running = 0.0
		running_kl = 0.0
		for (x,) in tqdm(train_loader, desc=f"VAE Epoch {epoch+1}/{cfg.epochs}"):
			x = x.to(device)
			recon, mu, logvar = model(x)
			loss, _, kl = vae_loss(recon, x, mu, logvar, beta=cfg.beta)
			opt.zero_grad()
			loss.backward()
			opt.step()
			running += loss.item() * x.size(0)
			running_kl += kl.item() * x.size(0)

		train_loss = running / len(train_set)
		train_kl = running_kl / len(train_set)
		train_losses.append(train_loss)
		train_kls.append(train_kl)

		model.eval()
		val_running = 0.0
		with torch.no_grad():
			for (x,) in val_loader:
				x = x.to(device)
				recon, mu, logvar = model(x)
				loss, _, _ = vae_loss(recon, x, mu, logvar, beta=cfg.beta)
				val_running += loss.item() * x.size(0)

		val_loss = val_running / max(1, len(val_set))
		val_losses.append(val_loss)
		print(f"Epoch {epoch+1}: train={train_loss:.4f}, kl={train_kl:.4f}, val={val_loss:.4f}")

	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"config": vars(cfg),
			"data_config": vars(data_cfg),
		},
		paths.checkpoints_dir / "vae.pt",
	)

	save_loss_curve(train_losses, paths.plots_dir / "vae_train_total_loss.png", "VAE Total Loss", "Loss")
	save_loss_curve(val_losses, paths.plots_dir / "vae_val_total_loss.png", "VAE Validation Loss", "Loss")
	save_loss_curve(train_kls, paths.plots_dir / "vae_train_kl_loss.png", "VAE KL Divergence", "KL")
	print(f"Saved checkpoint to {paths.checkpoints_dir / 'vae.pt'}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train VAE for multi-genre music generation.")
	parser.add_argument("--epochs", type=int, default=25)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--beta", type=float, default=0.1)
	parser.add_argument("--window-size", type=int, default=128)
	parser.add_argument("--stride", type=int, default=64)
	return parser.parse_args()


if __name__ == "__main__":
	train(parse_args())

