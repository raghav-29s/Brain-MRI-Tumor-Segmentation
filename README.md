"""# 🧠 Brain Tumor Segmentation using Deep Learning

<div align="center">
  
### Automatic pixel-level tumor segmentation from brain MRI scans
### using three deep learning architectures

</div>

---

## 📌 Overview

Brain tumor segmentation identifies **exactly which pixels** in an MRI scan belong to a tumor.

| Task | Output | Example |
|------|--------|---------|
| Classification | Label | "Tumor present" |
| Detection | Bounding box | Rectangle around tumor |
| **Segmentation** ← this project | Pixel mask | Exact tumor region highlighted |

This project implements and compares **3 deep learning architectures**, trains them on real MRI data, and deploys the best model as a **Streamlit web app**.

---

## 🏆 Results

| Model | Dice ↑ | IoU ↑ | Precision ↑ | Recall ↑ |
|-------|:------:|:-----:|:-----------:|:--------:|
| U-Net (baseline) | 0.5164 | 0.3758 | 0.4340 | 0.7582 |
| TransUNet | 0.6525 | 0.5264 | 0.6152 | 0.8097 |
| **ResNet50-UNet** ⭐ | **0.7351** | **0.6178** | **0.7610** | **0.7535** |

> 🚀 ResNet50-UNet achieved **42% improvement** in Dice Score over U-Net baseline using pretrained ImageNet encoder.

---

## 🧬 Architecture

### 1️⃣ U-Net — Baseline

```mermaid
flowchart TD
    A[Input MRI\\n3 × 128 × 128] --> B[Encoder Block 1\\nConv 64]
    B --> C[Encoder Block 2\\nConv 128]
    C --> D[Encoder Block 3\\nConv 256]
    D --> E[Encoder Block 4\\nConv 512]
    E --> F[Bottleneck\\nConv 1024]
    F --> G[Decoder Block 4\\nUpConv 512]
    G --> H[Decoder Block 3\\nUpConv 256]
    H --> I[Decoder Block 2\\nUpConv 128]
    I --> J[Decoder Block 1\\nUpConv 64]
    J --> K[Output Mask\\n1 × 128 × 128]

    B -- skip --> J
    C -- skip --> I
    D -- skip --> H
    E -- skip --> G

    style A fill:#4A90D9,color:#fff
    style F fill:#E67E22,color:#fff
    style K fill:#27AE60,color:#fff
```

**Key idea:** Skip connections pass encoder features directly to decoder — preserving fine spatial details lost during downsampling.

**Limitation:** CNN only sees local regions — misses global tumor shape context.

---

### 2️⃣ TransUNet — CNN + Vision Transformer

```mermaid
flowchart TD
    A[Input MRI\\n3 × 128 × 128] --> B[CNN Encoder\\nLocal features]
    B --> C[Split into Patches\\n8 × 8 grid]
    C --> D[Patch Embedding\\nTokens]
    D --> E[Self Attention Layer 1]
    E --> F[Self Attention Layer 2]
    F --> G[Self Attention Layer 3]
    G --> H[Self Attention Layer 4]
    H --> I[Self Attention Layer 5]
    I --> J[Self Attention Layer 6]
    J --> K[Reshape to Feature Map]
    K --> L[CNN Decoder\\nReconstruct mask]
    L --> M[Output Mask\\n1 × 128 × 128]

    B -- skip connections --> L

    style A fill:#4A90D9,color:#fff
    style E fill:#8E44AD,color:#fff
    style F fill:#8E44AD,color:#fff
    style G fill:#8E44AD,color:#fff
    style H fill:#8E44AD,color:#fff
    style I fill:#8E44AD,color:#fff
    style J fill:#8E44AD,color:#fff
    style M fill:#27AE60,color:#fff
```

**Key idea:** Transformer attends to every patch simultaneously — understands global tumor context that CNN misses.

**Result:** Fewer false positives, better boundary detection. Dice 0.5164 → 0.6525 (+26%)

---

### 3️⃣ ResNet50-UNet — Transfer Learning ⭐ Best

```mermaid
flowchart TD
    A[Input MRI\\n3 × 256 × 256] --> B[ResNet50 Layer 1\\nEdges and colors\\nPretrained]
    B --> C[ResNet50 Layer 2\\nShapes and curves\\nPretrained]
    C --> D[ResNet50 Layer 3\\nParts and regions\\nPretrained]
    D --> E[ResNet50 Layer 4\\nHigh-level features\\nPretrained]
    E --> F[Decoder Block 1\\nUpConv trained fresh]
    F --> G[Decoder Block 2\\nUpConv trained fresh]
    G --> H[Decoder Block 3\\nUpConv trained fresh]
    H --> I[Decoder Block 4\\nUpConv trained fresh]
    I --> J[Output Mask\\n1 × 256 × 256]

    B -- skip --> I
    C -- skip --> H
    D -- skip --> G
    E -- skip --> F

    style A fill:#4A90D9,color:#fff
    style B fill:#E74C3C,color:#fff
    style C fill:#E74C3C,color:#fff
    style D fill:#E74C3C,color:#fff
    style E fill:#E74C3C,color:#fff
    style J fill:#27AE60,color:#fff
```

**Key idea:** Encoder pretrained on 1.2M ImageNet images — already knows visual patterns. Only decoder trained from scratch on MRI data.

**Result:** Fastest convergence, best performance. Dice 0.5164 → 0.7351 (+42%)

---

### 🔄 End-to-End Pipeline

```mermaid
flowchart LR
    A[🗂️ Raw MRI Dataset\\n110 patients\\n3929 images] --> B[⚙️ Preprocessing\\nResize, Normalize\\nAugmentation]
    B --> C[📦 DataLoader\\nTrain 70%\\nVal 15% Test 15%]
    C --> D{Model Training}
    D --> E[U-Net\\n20 epochs]
    D --> F[TransUNet\\n50 epochs]
    D --> G[ResNet-UNet\\n50 epochs]
    E --> H[📊 Evaluation\\nDice IoU\\nPrecision Recall]
    F --> H
    G --> H
    H --> I[🏆 Best Model\\nResNet-UNet\\nDice 0.7351]
    I --> J[🖥️ Streamlit App\\nDeploy]

    style A fill:#3498DB,color:#fff
    style D fill:#E67E22,color:#fff
    style I fill:#27AE60,color:#fff
    style J fill:#9B59B6,color:#fff
```

---

## 📦 Dataset

**[LGG Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)** — Kaggle

- 110 patients, 3929 MRI slices
- Image size: 256×256 RGB
- Binary labels: tumor (1) vs background (0)
- Class split: 34% tumor / 66% no tumor
- Dataset split: 70% train / 15% val / 15% test

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
`OpenCV` `Google Colab` `NumPy`

---

"""

with open('/content/brain_tumor_segmentation/README.md', 'w') as f:
    f.write(readme_content)

print("✅ README.md written successfully")
print(f"✅ Total characters: {len(readme_content):,}")
