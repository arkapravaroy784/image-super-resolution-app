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

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load both models when the server starts
models = {
    2: edsr(scale=2, pretrained=True).to(device).eval(),
    4: edsr(scale=4, pretrained=True).to(device).eval(),
}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upscale")
async def upscale(file: UploadFile = File(...), scale: int = 2):
    if scale not in (2, 4):
        raise HTTPException(status_code=400, detail="scale must be 2 or 4")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()
    lr = to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr = models[scale](lr)

    sr_img = to_pil(sr.squeeze(0).clamp(0, 1).cpu())

    buf = io.BytesIO()
    sr_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")