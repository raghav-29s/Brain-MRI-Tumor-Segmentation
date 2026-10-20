"""# 🧠 Brain Tumor Segmentation using Deep Learning

<div align="center">


**Automatic pixel-level tumor segmentation from brain MRI scans using three deep learning architectures.**

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

### 1️⃣ U-Net — The Baseline

> Classic encoder-decoder with skip connections. Trained from scratch.


Input MRI (3 × 128 × 128)
│
▼
┌─────────────────────┐
│ ENCODER │ Extracts features, reduces spatial size
│ Conv(64) ──────────────────────────────────────┐
│ Conv(128) ──────────────────────────────────┐ │
│ Conv(256) ──────────────────────────────┐ │ │
│ Conv(512) ──────────────────────────┐ │ │ │
└────────────────────┬┘ │ │ │ │
▼ │ │ │ │
┌─────────────────────┐ │ │ │ │ skip
│ BOTTLENECK │ │ │ │ │ connections
│ Conv(1024) │ │ │ │ │
└────────────────────┬┘ │ │ │ │
▼ │ │ │ │
┌─────────────────────┐ │ │ │ │
│ DECODER │ │ │ │ │
│ UpConv(512) ◄──────────────────────┘ │ │ │
│ UpConv(256) ◄──────────────────────────┘ │ │
│ UpConv(128) ◄──────────────────────────────┘ │
│ UpConv(64) ◄──────────────────────────────────┘
└────────────────────┬┘
▼
Mask (1 × 128 × 128)


**Limitation:** CNN only sees small local regions — misses global tumor shape.

---

### 2️⃣ TransUNet — CNN + Vision Transformer

> Adds a Vision Transformer in the bottleneck to capture global context.

Input MRI (3 × 128 × 128)
│
▼
┌─────────────────────┐
│ CNN ENCODER │ Captures local features (edges, textures)
└────────┬────────────┘
│ skip connections
▼
┌─────────────────────────────────────────┐
│ VISION TRANSFORMER │
│ │
│ Feature map → Split into patches │
│ Patches → Sequence of tokens │
│ Tokens → Self-Attention (6 layers)│
│ │
│ Every patch attends to every other │
│ patch = understands GLOBAL context │
└────────┬────────────────────────────────┘
│
▼
┌─────────────────────┐
│ CNN DECODER │ Reconstructs spatial mask
└────────┬────────────┘
▼
Mask (1 × 128 × 128)


**Why Transformer helps:** CNN sees small windows. Transformer sees the whole image at once.
Result: fewer false positives, better boundary detection.

**Improvement:** 0.5164 → 0.6525 Dice (+26%)

---

### 3️⃣ ResNet50-UNet — Transfer Learning ⭐ Best Model

> Pretrained ResNet50 encoder — already knows 1.2M ImageNet images.

Input MRI (3 × 256 × 256)
│
▼
┌──────────────────────────────────────┐
│ ResNet50 ENCODER │
│ (Pretrained on ImageNet) │
│ │
│ Layer 1 → edges, colors │
│ Layer 2 → shapes, curves │ Already learned from
│ Layer 3 → parts, regions │ 1.2 million images
│ Layer 4 → high-level semantics │
│ │
│ 25M pretrained parameters │
└──────────────────┬───────────────────┘
│ skip connections
▼
┌──────────────────────────────────────┐
│ DECODER │
│ UpConv blocks (trained fresh) │ Learns medical context
│ + skip connections │ on top of ImageNet knowledge
└──────────────────┬───────────────────┘
▼
Mask (1 × 256 × 256)


**Why it wins:**

From scratch → learns from 762 images → limited knowledge
Pretrained → starts from 1.2M images → rich visual knowledge
adapted to medical imaging

**Improvement:** 0.5164 → 0.7351 Dice (+42%)

---

### 📊 Side-by-Side Comparison
                U-Net          TransUNet      ResNet50-UNet
                  │                │                │

Encoder From scratch From scratch ImageNet pretrained
Bottleneck Conv only Transformer ResNet blocks
Image size 128×128 128×128 256×256
Training data 20% 40% 100%
Epochs 20 50 50
│ │ │
Dice Score 0.5164 0.6525 0.7351 ✅
IoU Score 0.3758 0.5264 0.6178 ✅
Precision 0.4340 0.6152 0.7610 ✅
Recall 0.7582 0.8097 0.7535 ✅


---

## 📦 Dataset

**[LGG Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)** — Kaggle

110 patients
3929 MRI slices total
Image size: 256×256 RGB
Labels: tumor mask (binary)
Tumor: 34% | No tumor: 66%
Split: 70% train / 15% val / 15% test


---

## 🗂️ Project Structure

brain-tumor-segmentation/
│
├── src/
│ ├── dataset.py ← PyTorch Dataset + augmentation pipeline
│ ├── models.py ← UNet, TransUNet, ResNet50-UNet definitions
│ └── metrics.py ← Dice, IoU, Precision, Recall functions
│
├── models/ ← saved model weights
│ ├── best_unet.pth
│ ├── best_transunet.pth
│ └── best_smp_unet.pth
│
├── outputs/ ← prediction visualizations + results
│ ├── unet_predictions.png
│ ├── transunet_predictions.png
│ └── final_results.json
│
├── app.py ← Streamlit web application
├── requirements.txt ← all dependencies
└── README.md





---

## 🔮 Future Work

- [ ] Train on full BraTS 2020 dataset (25,000 slices)
- [ ] Multi-class segmentation (tumor core, enhancing, edema)
- [ ] 3D volumetric segmentation
- [ ] Deploy on HuggingFace Spaces
- [ ] Add GradCAM visualizations for explainability
- [ ] Test with EfficientNet and ViT encoders

---

## 📄 References

- [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [LGG Segmentation Dataset — Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)

---

## 🛠️ Tech Stack

`PyTorch` `segmentation-models-pytorch` `albumentations`
`einops` `OpenCV` `Streamlit` `Google Colab` `NumPy`

---

## 👤 Author

**Raghav**
GitHub: [@raghav-29s](https://github.com/raghav-29s)

---

<div align="center">

⭐ **Star this repo if you found it helpful!**

</div>
"""

with open('/content/brain_tumor_segmentation/README.md', 'w') as f:
    f.write(readme_content)

print("✅ README.md written successfully")
print(f"✅ Total characters: {len(readme_content):,}")
