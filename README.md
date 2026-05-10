# Sketch-to-Image Generation using GANs and Diffusion Models

> EE655 Course Project — IIT Kanpur  
> Novel Approaches Tackling Sketch-to-Image Generation

---

## Overview

This project explores multiple deep learning approaches for converting hand-drawn sketches into realistic images.

We compare and implement:

- CLIP-Guided Stable Diffusion + ControlNet
- CycleGAN
- Custom Pix2Pix-style Conditional GAN (cGAN)

The objective is to study different paradigms of sketch-to-image synthesis and evaluate them in terms of:

- Image quality
- Semantic alignment
- Training efficiency
- Flexibility
- Fine-grained controllability

The project also proposes hybrid pipelines combining pretrained diffusion models with controllable sketch-conditioning techniques.

---

# Features

- Sketch-to-image generation using ControlNet + Stable Diffusion
- Custom Pix2Pix-inspired GAN architecture
- CycleGAN for unpaired image translation
- Automatic sketch generation using:
  - Canny Edge Detection
  - Thresholding
  - Photoshop-style filtering
- Prompt enhancement using CLIP
- Synthetic dataset generation for domains lacking paired data
- Comparative study between GANs and diffusion models

---

# Project Architecture

## 1. CLIP-Guided Stable Diffusion Pipeline

### Step 1 — Sketch to Prompt using CLIP
- Extracts semantic meaning from sketches
- Converts sketches into descriptive prompts

### Step 2 — Prompt Refinement
Enhances prompts using:
- Style modifiers
- Lighting and shadows
- Composition enhancement
- Quality boosters

### Step 3 — Image Generation
Uses:
- ControlNet
- Stable Diffusion

Supported conditioning:
- Canny edges
- Scribble
- SoftEdge / SoftNoise

This preserves:
- Structure
- Spatial layout
- Contours

while generating realistic textures and lighting.

---

## 2. CycleGAN Approach

CycleGAN is used for:
- Sketch → Image translation
- Image → Sketch translation

Advantages:
- Works with unpaired datasets
- Useful for semi-supervised data augmentation

---

## 3. Custom GAN (Pix2Pix-style)

Custom implementation inspired by Pix2Pix using:
- U-Net Generator
- PatchGAN Discriminator
- Custom upsampling/downsampling blocks

### Generator Loss
Combination of:
- Adversarial Loss
- L1 Loss

### Training Details

| Parameter | Value |
|---|---|
| Image Size | 128x128 |
| Batch Size | 4 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning Rate | 2e-4 |

---

# Tech Stack

## Deep Learning Frameworks
- TensorFlow / Keras
- PyTorch

## Models & Libraries
- Stable Diffusion
- ControlNet
- CLIP
- CycleGAN
- Pix2Pix
- OpenCV

## Utilities
- NumPy
- Matplotlib
- PIL

---

# Dataset

## Primary Dataset
Human Faces Dataset:
- https://www.kaggle.com/datasets/ashwingupta3012/human-faces

## Additional Dataset
Sketchy Dataset:
- https://sketchy.eye.gatech.edu/

---

# Dataset Generation Strategy

Since paired sketch-image datasets are limited, synthetic sketch-image pairs were generated using:

- Canny Edge Detection
- Thresholding
- Photoshop filters
- Image preprocessing pipelines

This significantly improved training diversity and robustness.

---

# Results

## ControlNet + Stable Diffusion

### Advantages
- High-quality realistic outputs
- Highly flexible
- Works across domains
- Requires minimal retraining

### Limitations
- Slower inference
- Less fine-grained structural control
- Prompt-sensitive

---

## GAN-Based Models

### Advantages
- Faster inference
- Better domain-specific control
- Lightweight deployment

### Limitations
- Requires large datasets
- Resource-intensive training
- Less flexible across domains

---

# Comparison Table

| Feature | ControlNet + Diffusion | GAN-Based Models |
|---|---|---|
| Training Stability | High | Moderate |
| Inference Speed | Slow | Fast |
| Output Quality | Very High | High |
| Flexibility | Excellent | Moderate |
| Data Requirement | Low (fine-tuning) | High |
| Modularity | High | Lower |
| User Control | Prompt-based | Architecture-based |

---

# Repository Structure

```bash
.
├── datasets/
├── models/
│   ├── cyclegan/
│   ├── custom_gan/
│   └── diffusion/
├── notebooks/
├── outputs/
├── scripts/
├── utils/
├── train.py
├── inference.py
└── README.md
