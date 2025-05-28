import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp

st.set_page_config(page_title="Anorectal Lesion Segmentation", layout="wide")
st.title("🧠 AI-Based Anorectal Lesion Segmentation")

# Model yükle
model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=1)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

uploaded_file = st.file_uploader("📤 Upload an MRI slice (.png)", type=["png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image_resized = image.resize((256, 256))
    image_np = np.array(image_resized) / 255.0
    input_tensor = torch.tensor(image_np).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        output = model(input_tensor)
        output_mask = torch.sigmoid(output).squeeze().numpy()
        pred_mask = (output_mask > 0.5).astype(np.uint8) * 255

    # MRI gri görüntü
    original = np.array(image_resized).astype(np.uint8)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # 🔵 1. Maske yumuşatma (Gaussian blur)
    pred_mask_blurred = cv2.GaussianBlur(pred_mask, (7, 7), 0)

    # 🔴 2. Sade kırmızı renkli maske oluştur
    mask_rgb = np.zeros_like(original_rgb)
    mask_rgb[:, :, 2] = pred_mask_blurred  # Red channel

    # 🟢 3. Overlay: Saydam karışım (daha şeffaf)
    overlay = cv2.addWeighted(original_rgb, 0.85, mask_rgb, 0.15, 0)

    # Görsel yerleşimi
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original MRI")
        st.image(original, use_column_width=True, channels="L")

    with col2:
        st.subheader("Segmentation Mask")
        st.image(pred_mask, use_column_width=True, clamp=True)

    with col3:
        st.subheader("Overlay on MRI")
        st.image(overlay, use_column_width=True, channels="BGR")

    # Alan bilgisi
    lesion_area = np.sum(pred_mask > 0)
    st.markdown(f"### 📐 Lesion Area: {lesion_area} pixels")
