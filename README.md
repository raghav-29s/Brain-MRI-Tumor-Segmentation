"""# 🧠 Brain Tumor Segmentation using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-orange?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Automatic pixel-level tumor segmentation from brain MRI scans using three deep learning architectures.**

[Demo](#streamlit-app) • [Results](#results) • [Setup](#setup) • [Architecture](#architecture)

</div>

---

## 📌 Overview

This project implements and compares three deep learning models for brain tumor segmentation:

- **U-Net** — classic encoder-decoder with skip connections (baseline)
- **TransUNet** — U-Net enhanced with Vision Transformer in the bottleneck
- **ResNet50-UNet** — U-Net with pretrained ImageNet encoder (best performance)

The models are trained on the **LGG Brain MRI Segmentation dataset** and deployed via a **Streamlit web app** where you can upload any brain MRI and instantly get a tumor mask prediction.

---

## 🏆 Results

| Model | Dice Score | IoU | Precision | Recall |
|-------|:----------:|:---:|:---------:|:------:|
| U-Net (scratch) | 0.5164 | 0.3758 | 0.4340 | 0.7582 |
| TransUNet | 0.6525 | 0.5264 | 0.6152 | 0.8097 |
| **ResNet50-UNet** ⭐ | **0.7351** | **0.6178** | **0.7610** | **0.7535** |

> 🚀 ResNet50-UNet with pretrained ImageNet encoder achieved **42% improvement** in Dice Score over the U-Net baseline.

---

## 🧬 Architecture

### U-Net
