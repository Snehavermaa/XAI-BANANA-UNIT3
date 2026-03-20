# README for Banana CNN Visualization Notebook

```markdown
# Banana CNN Visualization: Saliency Maps & Activation Maximization

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
git clone https://github.com/Snehavermaa/XAI-BANANA-UNIT3.git
cd XAI-BANANA-UNIT3

# Install dependencies
pip install torch torchvision matplotlib numpy opencv-python pillow requests
```

### Dependencies
- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- OpenCV
- PIL
- requests

## 📓 Notebook Structure

```
PES1UG23AM309_Banana_U3.ipynb
├── 1. Setup & Imports
├── 2. Task 1: Vanilla Gradient Saliency
│   ├── 2.1 Create Synthetic Banana
│   ├── 2.2 Load Pretrained ResNet18
│   ├── 2.3 Compute Vanilla Gradients
│   ├── 2.4 Apply SmoothGrad
│   └── 2.5 Visualize & Interpret
├── 3. Task 2: Activation Maximization
│   ├── 3.1 Class Definition
│   ├── 3.2 Basic Configuration
│   ├── 3.3 Aggressive Regularization
│   ├── 3.4 Minimal Regularization
│   └── 3.5 Compare All Techniques
└── 4. Conclusions & Interpretation
```

## 🎨 Sample Results

### Task 1: Saliency Maps
```
Original Image    →    Vanilla Gradient    →    SmoothGrad
   [Banana]              [Edge Heatmap]         [Clean Heatmap]
```

**Interpretation:** 
- Vanilla gradients highlight edges and contours (noisy)
- SmoothGrad produces cleaner, more coherent regions
- Both confirm the model focuses on banana shape and spots

### Task 2: Activation Maximization
```
Random Noise    →    After 500 iterations    →    Final Pattern
   [Noise]              [Yellow Curves]           [Banana Features]
```

**Interpretation:**
- Generated images show yellow curved textures
- Reveals CNN's texture bias over shape
- Regularization crucial for interpretable results

## 🔍 Key Findings

### From Task 1:
- ✅ Saliency maps highlight edges more than object interiors
- ✅ Vanilla gradients are noisy due to gradient saturation
- ✅ SmoothGrad effectively reduces noise
- ✅ The model attends to distinctive features (brown spots)

### From Task 2:
- ✅ CNN strongly prefers yellow/orange colors for bananas
- ✅ Texture patterns dominate over global shape
- ✅ Regularization is essential for meaningful visualizations
- ✅ Different regularization produces varying interpretability

## 📈 Regularization Comparison

| Method | Result | Best For |
|--------|--------|----------|
| No Regularization | Noisy, high-frequency | Exploring model weaknesses |
| L2 Only | Bounded values | Preventing extreme pixels |
| TV Only | Smooth transitions | Coherent regions |
| Blur Only | Soft features | Spatial coherence |
| All Combined | Balanced | Most interpretable |

## 🚦 Running the Notebook

### Option 1: Local Jupyter
```bash
jupyter notebook "Banana CNN Visualization.ipynb"
```

### Option 2: Google Colab
Click the "Open in Colab" badge at the top of this README

## 📝 Customization

You can easily modify the notebook to:
- Try different ImageNet classes (change `banana_class_idx = 954`)
- Use real images instead of synthetic
- Adjust regularization parameters
- Increase optimization iterations
- Try different base models (VGG, AlexNet, etc.)

## 🤝 Contributing

Contributions are welcome! Here are some ideas:
- Add more visualization techniques (Guided Backprop, GradCAM)
- Implement frequency-based regularization
- Add video processing capabilities
- Create interactive visualizations with Plotly

## 📚 Resources

- [Deep Inside Convolutional Networks: Visualising Image Classification Models](https://arxiv.org/abs/1312.6034)
- [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
- [ImageNet Class Index](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ImageNet for the pretrained model weights
- PyTorch team for the excellent deep learning framework
- The XAI research community for foundational papers

## 📧 Contact

For questions or feedback:
- Create an issue in this repository
- Contact: mesnehaverma23@gmail.com

---

**If you find this notebook useful, please give it a star!** ⭐
```
