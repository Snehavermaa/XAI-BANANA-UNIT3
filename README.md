# 🍌 Banana CNN Visualization: Saliency Maps & Activation Maximization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/banana-cnn-visualization/blob/main/banana_visualization.ipynb)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This notebook explores **Explainable AI (XAI)** techniques for computer vision by visualizing how a pretrained ResNet18 model identifies bananas. It implements two fundamental visualization techniques:

1. **Task 1: Vanilla Gradient Saliency Maps** - Shows which parts of an image influence the model's decision
2. **Task 2: Activation Maximization** - Generates synthetic images that maximally activate the banana class neuron

## 🎯 Objectives

### Task 1: Vanilla Gradient Saliency Maps
- Understand which image regions contribute to banana classification
- Visualize gradients of the banana class score with respect to input pixels
- Compare vanilla gradients with SmoothGrad for noise reduction
- Analyze whether the model focuses on edges, textures, or object regions

### Task 2: Activation Maximization
- Generate synthetic images from scratch that "fool" the model into seeing bananas
- Reveal what features the banana class neuron has learned (texture bias, color preference)
- Experiment with different regularization techniques (L2, Total Variation, Gaussian blur)
- Understand how CNNs build internal representations of objects

## 🚀 Techniques Implemented

| Technique | Description | Regularization |
|-----------|-------------|----------------|
| **Vanilla Gradient** | Basic gradient computation | None |
| **SmoothGrad** | Average gradients over noisy samples | Noise averaging |
| **Activation Maximization** | Gradient ascent from random noise | L2, TV, Blur |

## 📊 Key Visualizations

### Task 1 Outputs:
- Original image with synthetic banana
- Vanilla gradient heatmap (noisy, edge-focused)
- SmoothGrad heatmap (cleaner, region-focused)
- Edge detection comparison
- Saliency distribution histograms

### Task 2 Outputs:
- Generated images from 3 regularization configurations
- Class score progression over iterations
- Color distribution analysis
- Frequency spectrum analysis
- Comparison of 8 regularization combinations

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/banana-cnn-visualization.git
cd banana-cnn-visualization

# Install dependencies
pip install torch torchvision matplotlib numpy opencv-python pillow requests
