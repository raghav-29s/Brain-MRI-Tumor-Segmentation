
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append("src")
from models import UNet, TransUNet, get_resnet_unet

st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Segmentation")
st.markdown("Upload a brain MRI image to detect and segment tumors automatically.")

# ── Sidebar ──────────────────────────────────────
st.sidebar.title("Settings")
model_choice = st.sidebar.radio(
    "Select Model",
    ["ResNet-UNet (Best)", "TransUNet", "U-Net (Baseline)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.markdown("""
| Model | Dice |
|-------|------|
| ResNet-UNet | 0.7351 |
| TransUNet   | 0.6525 |
| U-Net       | 0.5164 |
""")

threshold = st.sidebar.slider(
    "Prediction threshold", 0.1, 0.9, 0.5, 0.05)

# ── Load model ───────────────────────────────────
@st.cache_resource
def load_model(choice):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if choice == "U-Net (Baseline)":
        model = UNet()
        model.load_state_dict(
            torch.load("models/best_unet.pth",
                       map_location=device))
    elif choice == "TransUNet":
        model = TransUNet()
        model.load_state_dict(
            torch.load("models/best_transunet.pth",
                       map_location=device))
    else:
        model = get_resnet_unet()
        model.load_state_dict(
            torch.load("models/best_smp_unet.pth",
                       map_location=device))
    model.eval()
    return model.to(device), device

model, device = load_model(model_choice)

# ── Transform ────────────────────────────────────
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# ── Upload ───────────────────────────────────────
uploaded = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg","jpeg","png","tif","tiff"]
)

if uploaded:
    # Read image
    file_bytes = np.asarray(
        bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = img.copy()

    # Preprocess
    aug    = transform(image=img)
    tensor = aug["image"].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)
        mask = (pred > threshold).float()
        mask = mask.squeeze().cpu().numpy()

    # Resize mask to original size
    mask_resized = cv2.resize(
        mask,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Create overlay
    overlay = original.copy()
    overlay[mask_resized > 0] = [255, 50, 50]
    overlay = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)

    # Tumor detected?
    tumor_detected = mask_resized.sum() > 100
    if tumor_detected:
        st.error("🔴 Tumor Detected")
    else:
        st.success("🟢 No Tumor Detected")

    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**MRI Input**")
        st.image(original, use_column_width=True)
    with col2:
        st.markdown("**Predicted Mask**")
        st.image(mask_resized, use_column_width=True,
                 clamp=True)
    with col3:
        st.markdown("**Overlay**")
        st.image(overlay, use_column_width=True)

    # Stats
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        tumor_pct = mask_resized.mean() * 100
        st.metric("Tumor Area", f"{tumor_pct:.2f}%")
    with col2:
        st.metric("Model Used", model_choice.split()[0])
    with col3:
        st.metric("Threshold", f"{threshold:.2f}")
else:
    st.info("👆 Upload a brain MRI image to get started")
    st.markdown("""
    ### How it works
    1. Upload a brain MRI scan
    2. Select a model from the sidebar
    3. The model segments the tumor automatically
    4. View the predicted mask and overlay
    """)
