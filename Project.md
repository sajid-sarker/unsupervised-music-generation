# Unsupervised Neural Network for Multi-Genre Music Generation

### 1\. Project Motivation

Music is a structured temporal signal containing complex patterns such as:  
• Melody progression  
• Harmony and chords  
• Rhythm and tempo variation  
• Genre-specific style distributions

Traditional supervised learning requires labeled music data, which is expensive. Therefore, this project focuses on unsupervised generative neural networks that can learn musical representations without explicit genre labels.

Goal: Build a deep unsupervised model capable of generating novel music pieces across multiple genres such as Classical, Jazz, Rock, Pop, and Electronic.

### 2\. Problem Definition

Let a music sequence be represented as:

X \= {x1, x2, ..., xT }

where each xt is a symbolic event such as:

• note-on event  
• note-off event  
• velocity  
• duration

The objective is to learn an unsupervised generative distribution: pθ(X) such that the model can sample: Xˆ ∼ pθ(X) to generate realistic genre-consistent music.

### 3\. Dataset Requirements

Students must use at least one publicly available MIDI dataset listed below:

* MAESTRO Dataset \- Genre: Classical Piano   
* Lakh MIDI Dataset \- Genre: Multi-Genre Collection   
* Groove MIDI Dataset \- Genre: Jazz / Drums / Rhythm

Preprocessing Pipeline:

1\. Convert MIDI into piano-roll or token-based representation  
2\. Normalize timing resolution (e.g., 16 steps per bar)  
3\. Segment sequences into fixed-length windows  
4\. Model Tasks and Mathematical Formulation

### Task 1 (Easy): LSTM Autoencoder Music Generator

**Goal:** Implement a basic unsupervised LSTM Autoencoder that reconstructs and generates  
short music sequences from a single genre.

**Mathematical Model:**  
Encoder learns latent embedding: z \= fφ(X)

Decoder reconstructs: Xˆ \= gθ(z)

Loss Function: Sum of Residual squares

**Deliverables:**  
• Autoencoder implementation code  
• Reconstruction loss curve  
• 5 generated MIDI samples

### Task 2 (Medium): Variational Autoencoder (VAE) Multi-Genre Generator

**Goal**  
Extend Task 1 into a VAE to generate more diverse music across multiple genres.

**Latent Distribution**

qφ(z|X) \= N (μ(X), σ(X))

Sampling: z \= μ \+ σ ⊙ ε, ε ∼ N (0, I)

**Loss Function**

L(VAE) \= Lrecon \+ β . D(KL)(qφ(z|X)∥p(z))

**Deliverables**  
• VAE code with KL-divergence loss  
• Multi-genre generation outputs (8 samples)  
• Metric comparison vs Task 1

### Task 3 (Hard): Transformer-Based Music Generator

**Goal**  
Develop a Transformer decoder capable of generating long coherent sequences. 

**Autoregressive Probability**  
p(X) \= Product (p(xt|x\<t); t=1, T

**Training Loss:**  
Summation of log pθ(xt|x\<t)

**Perplexity Metric**  
Perplexity \= exp (1/T . L(TR))

**Deliverables**  
• Transformer architecture implementation  
• Perplexity evaluation report  
• 10 long-sequence generated compositions

