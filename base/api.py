import os
os.environ["TORCH_HOME"] = "/tmp/torch"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
from torchsr.models import edsr
from PIL import Image
import torchvision.transforms as T
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Force CPU (Railway does not provide GPU)
device = "cpu"

# Lazy-loaded models cache
models = {}

def get_model(scale):
    if scale not in models:
        print(f"[INFO] Loading model for scale {scale}x...")
        try:
            models[scale] = edsr(scale=scale, pretrained=True, progress=False).to(device).eval()
        except Exception as e:
            print(f"[ERROR] Model loading failed: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")
    return models[scale]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upscale")
async def upscale(file: UploadFile = File(...), scale: int = 2):
    # Validate scale
    if scale not in (2, 4):
        raise HTTPException(status_code=400, detail="scale must be 2 or 4")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read file
    contents = await file.read()

    # Safe image loading
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Transform
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    lr = to_tensor(img).unsqueeze(0).to(device)

    # Lazy load model
    model = get_model(scale)

    # Inference
    with torch.no_grad():
        sr = model(lr)

    # Convert back to image
    sr_img = to_pil(sr.squeeze(0).clamp(0, 1).cpu())

    # Return as streaming response
    buf = io.BytesIO()
    sr_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
