import os
os.environ["TORCH_HOME"] = "/tmp/torch"
import streamlit as st
import torch
from torchsr.models import edsr
from PIL import Image
import torchvision.transforms as T
from streamlit_image_comparison import image_comparison
import io

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Image Super-Resolution",
    layout="centered"
)

st.title("🔍 Image Super-Resolution App")
st.write("Upscale images using deep learning (EDSR)")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    return {
        2: edsr(scale=2, pretrained=True).to(device).eval(),
        4: edsr(scale=4, pretrained=True).to(device).eval(),
    }

models = load_models()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

scale = st.radio(
    "Select upscale factor",
    [2, 4],
    horizontal=True
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(img, use_container_width=True)

    # ---------------- INFERENCE ----------------
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    lr = to_tensor(img).unsqueeze(0).to(device)
    model = models[scale]

    with torch.no_grad():
        sr = model(lr)

    sr_img = to_pil(sr.squeeze(0).clamp(0, 1).cpu())

    # ---------------- COMPARISON SLIDER ----------------
    st.subheader("Before / After Comparison")

    image_comparison(
        img1=img,
        img2=sr_img,
        label1="Before (Low Resolution)",
        label2=f"After ({scale}× Super-Resolution)"
    )

    # ---------------- DOWNLOAD ----------------
    buf = io.BytesIO()
    sr_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="⬇️ Download Super-Resolved Image",
        data=byte_im,
        file_name=f"SR_x{scale}.png",
        mime="image/png"
    )

    # ---------------- INFO ----------------
    st.caption(
        f"Input size: {img.size[0]}×{img.size[1]} | "
        f"Output size: {sr_img.size[0]}×{sr_img.size[1]}"
    )
