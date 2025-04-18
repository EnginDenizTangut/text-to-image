# Text-to-Flower Image Generation with GANs

This project implements a Generative Adversarial Network (GAN) that generates flower images from text descriptions. The model is trained on a dataset of real flower images paired with their textual labels.

## Key Features
- Conditional GAN architecture for text-to-image generation
- GloVe-based text embeddings for semantic conditioning
- 128x128 RGB image generation
- PyTorch implementation with CUDA support

## Requirements
- Python 3.7+
- PyTorch 1.8+
- torchvision
- Pillow
- NumPy

## Installation
```bash
pip install torch torchvision pillow numpy
```

Model Overview
Generator Input:

Random noise vector (z) of shape [batch_size, 100]

Text embedding of shape [batch_size, 300]

Discriminator Input:

Image (real or generated), flattened to [batch_size, 3*128*128]

Loss function: Binary Cross-Entropy (BCE)

To Improve
Replace random embeddings with real GloVe vectors

Use a convolutional architecture (e.g., DCGAN)

Add text encoder (e.g., RNN or Transformer)

Use more detailed class labels or captions

License
This project is licensed under the MIT License.

References
Oxford 102 Flowers Dataset

GloVe Embeddings

GANs - Goodfellow et al.

PyTorch Documentation
