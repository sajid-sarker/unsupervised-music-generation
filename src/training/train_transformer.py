from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from src.config import DataConfig, PathConfig, TransformerConfig, ensure_output_dirs, get_device
from src.evaluation.metrics import save_loss_curve
from src.models.transformer import CausalTransformer
from src.preprocessing.midi_parser import build_piano_roll_windows, save_windows
from src.preprocessing.tokenizer import SimplePitchTokenizer, segment_token_sequence


def build_token_windows(paths: PathConfig, data_cfg: DataConfig, token_window: int, token_stride: int) -> np.ndarray:
	token_path = paths.processed_dir / "token_windows.npy"
	if token_path.exists():
		return np.load(token_path)

	roll_path = paths.processed_dir / "piano_roll_windows.npy"
	if roll_path.exists():
		roll_windows = np.load(roll_path)
	else:
		roll_windows = build_piano_roll_windows(paths.raw_midi_dir, data_cfg)
		save_windows(roll_windows, roll_path)

	tok = SimplePitchTokenizer()
	chunks = []
	for roll in roll_windows:
		tokens = tok.piano_roll_to_tokens(roll)
		tokens = tok.add_special_tokens(tokens)
		windows = segment_token_sequence(tokens, window_size=token_window, stride=token_stride)
		chunks.append(windows)

	if not chunks:
		return np.zeros((0, token_window), dtype=np.int64)

	all_windows = np.concatenate(chunks, axis=0).astype(np.int64)
	np.save(token_path, all_windows)
	return all_windows


def train(args: argparse.Namespace) -> None:
	paths = PathConfig()
	ensure_output_dirs(paths)

	dcfg = DataConfig(window_size=args.roll_window_size, stride=args.roll_stride)
	cfg = TransformerConfig(
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		max_seq_len=args.token_window_size,
	)
	tok = SimplePitchTokenizer()

	token_windows = build_token_windows(paths, dcfg, args.token_window_size, args.token_stride)
	if len(token_windows) == 0:
		raise RuntimeError("No token windows found. Add MIDI files to data/raw_midi first.")

	x = torch.tensor(token_windows[:, :-1], dtype=torch.long)
	y = torch.tensor(token_windows[:, 1:], dtype=torch.long)
	dataset = TensorDataset(x, y)
	train_size = int(0.9 * len(dataset))
	val_size = len(dataset) - train_size
	train_set, val_set = random_split(dataset, [train_size, val_size])

	train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
	val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

	device = get_device()
	model = CausalTransformer(
		vocab_size=cfg.vocab_size,
		d_model=cfg.d_model,
		nhead=cfg.nhead,
		num_layers=cfg.num_layers,
		dim_feedforward=cfg.dim_feedforward,
		dropout=cfg.dropout,
		max_seq_len=cfg.max_seq_len,
	).to(device)
	opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

	train_losses = []
	val_losses = []
	val_perplexities = []

	for epoch in range(cfg.epochs):
		model.train()
		train_running = 0.0
		for xb, yb in tqdm(train_loader, desc=f"TR Epoch {epoch+1}/{cfg.epochs}"):
			xb = xb.to(device)
			yb = yb.to(device)

			logits = model(xb)
			loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), yb.reshape(-1), ignore_index=tok.pad_token)
			opt.zero_grad()
			loss.backward()
			opt.step()
			train_running += loss.item() * xb.size(0)

		train_loss = train_running / len(train_set)
		train_losses.append(train_loss)

		model.eval()
		val_running = 0.0
		with torch.no_grad():
			for xb, yb in val_loader:
				xb = xb.to(device)
				yb = yb.to(device)
				logits = model(xb)
				loss = F.cross_entropy(
					logits.reshape(-1, cfg.vocab_size),
					yb.reshape(-1),
					ignore_index=tok.pad_token,
				)
				val_running += loss.item() * xb.size(0)

		val_loss = val_running / max(1, len(val_set))
		val_losses.append(val_loss)
		val_ppl = float(np.exp(val_loss))
		val_perplexities.append(val_ppl)
		print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, ppl={val_ppl:.2f}")

	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"config": vars(cfg),
			"data_config": vars(dcfg),
		},
		paths.checkpoints_dir / "transformer.pt",
	)

	save_loss_curve(train_losses, paths.plots_dir / "transformer_train_loss.png", "Transformer Train CE Loss", "Loss")
	save_loss_curve(val_losses, paths.plots_dir / "transformer_val_loss.png", "Transformer Validation CE Loss", "Loss")
	save_loss_curve(
		val_perplexities,
		paths.plots_dir / "transformer_val_perplexity.png",
		"Transformer Validation Perplexity",
		"Perplexity",
	)
	print(f"Saved checkpoint to {paths.checkpoints_dir / 'transformer.pt'}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train transformer-based music generator.")
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--roll-window-size", type=int, default=128)
	parser.add_argument("--roll-stride", type=int, default=64)
	parser.add_argument("--token-window-size", type=int, default=129)
	parser.add_argument("--token-stride", type=int, default=64)
	return parser.parse_args()


if __name__ == "__main__":
	train(parse_args())

