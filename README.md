# Unsupervised Neural Network for Multi-Genre Music Generation

A project for the Spring 2026 semester of CSE425 at BRACU

Goal: Build a deep unsupervised model capable of generating novel music pieces across multiple genres such as Classical, Jazz, Rock, Pop, and Electronic.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Put MIDI files under `data/raw_midi/`.

3. Train models:

```bash
python src/training/train_ae.py
python src/training/train_vae.py
python src/training/train_transformer.py
```

4. Generate MIDI samples:

```bash
python src/generation/generate_music.py --model ae --num-samples 5
python src/generation/generate_music.py --model vae --num-samples 8
python src/generation/generate_music.py --model transformer --num-samples 10 --seq-len 256
```

Outputs are saved in `outputs/checkpoints/`, `outputs/plots/`, and `outputs/generated_midis/`.
