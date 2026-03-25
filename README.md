# Image Super-Resolution API

A REST API for upscaling images using deep learning (EDSR model). Built with FastAPI and PyTorch.

**Live API URL:** `https://image-super-resolution-app.onrender.com`

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Endpoints](#api-endpoints)
- [Request & Response Examples](#request--response-examples)
- [Using the API in JavaScript](#using-the-api-in-javascript)
- [Using the API in Python](#using-the-api-in-python)
- [Error Handling](#error-handling)
- [Important Notes](#important-notes)

---

## Overview

This API takes an image uploaded by the user and returns a super-resolved (upscaled) version of it using the **EDSR (Enhanced Deep Super-Resolution)** model. The user can choose between **2x** or **4x** upscaling.

| Input | Output |
|-------|--------|
| Any PNG / JPG / JPEG image | Upscaled PNG image |
| Scale factor: 2 or 4 | Image dimensions multiplied by the scale factor |

---

## Project Structure

```
├── app.py               # Original Streamlit app (local UI)
├── api.py               # FastAPI backend (this API)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

- `app.py` — the original local Streamlit app with a UI. Run this if you want to use the app on your own machine.
- `api.py` — the API server. This is what is deployed on Render and what your frontend should talk to.

---

## How It Works

```
User sends image
      │
      ▼
POST /upscale?scale=2
      │
      ▼
FastAPI receives the image file
      │
      ▼
Image is converted to a PyTorch tensor
      │
      ▼
EDSR model runs inference (2x or 4x)
      │
      ▼
Output tensor is converted back to a PNG image
      │
      ▼
PNG image is returned directly in the response
```

### Model Loading

The EDSR models are loaded **lazily** — meaning a model is only loaded into memory the first time it is requested. This keeps memory usage low on the free server tier.

- `edsr(scale=2)` — loaded on first 2x request
- `edsr(scale=4)` — loaded on first 4x request

Once loaded, the model stays in memory for all future requests.

---

## API Endpoints

### `GET /health`

Check if the API is running.

| Property | Value |
|----------|-------|
| Method | `GET` |
| URL | `/health` |
| Auth required | No |

**Response:**
```json
{
  "status": "ok"
}
```

---

### `POST /upscale`

Upload an image and get back the upscaled version.

| Property | Value |
|----------|-------|
| Method | `POST` |
| URL | `/upscale` |
| Content-Type | `multipart/form-data` |
| Auth required | No |

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `scale` | integer | No | `2` | Upscale factor. Must be `2` or `4` |

**Request Body (form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | The image to upscale (PNG, JPG, or JPEG) |

**Response:**

Returns the upscaled image directly as a PNG file (`image/png`).

---

## Request & Response Examples

### Using cURL

**2x upscale:**
```bash
curl -X POST "https://image-super-resolution-app.onrender.com/upscale?scale=2" \
  -F "file=@your_image.jpg" \
  --output upscaled_image.png
```

**4x upscale:**
```bash
curl -X POST "https://image-super-resolution-app.onrender.com/upscale?scale=4" \
  -F "file=@your_image.jpg" \
  --output upscaled_image.png
```

---

## Using the API in JavaScript

```javascript
async function upscaleImage(imageFile, scale = 2) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch(
    `https://image-super-resolution-app.onrender.com/upscale?scale=${scale}`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }

  // Convert the response to a blob and create a URL
  const blob = await response.blob();
  const imageUrl = URL.createObjectURL(blob);

  return imageUrl; // Use this URL in an <img> tag
}

// Example usage with an <input type="file"> element
const fileInput = document.getElementById("fileInput");
fileInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  const upscaledUrl = await upscaleImage(file, 2);

  const img = document.getElementById("result");
  img.src = upscaledUrl;
});
```

---

## Using the API in Python

```python
import requests

def upscale_image(image_path, scale=2, output_path="upscaled.png"):
    url = f"https://image-super-resolution-app.onrender.com/upscale?scale={scale}"

    with open(image_path, "rb") as f:
        response = requests.post(url, files={"file": f})

    if response.status_code == 200:
        with open(output_path, "wb") as out:
            out.write(response.content)
        print(f"Saved upscaled image to {output_path}")
    else:
        print(f"Error: {response.json()}")

# Example usage
upscale_image("my_photo.jpg", scale=4, output_path="my_photo_4x.png")
```

---

## Error Handling

The API returns standard HTTP status codes with a JSON error message.

| Status Code | Meaning | Example |
|-------------|---------|---------|
| `200` | Success | Upscaled image returned |
| `400` | Bad request | Invalid scale value or non-image file uploaded |
| `422` | Validation error | Required field missing |
| `500` | Server error | Something went wrong internally |

**Example error response:**
```json
{
  "detail": "scale must be 2 or 4"
}
```

---

## Important Notes

### Cold Start Delay
This API is hosted on Render's free tier. If the API has not been used for a while, the server goes to sleep. The **first request after inactivity may take 50+ seconds** to respond while the server wakes up. All subsequent requests will be fast.

### Supported Image Formats
- PNG
- JPG / JPEG

### Interactive Testing
You can test all endpoints directly in the browser without writing any code by visiting:

```
https://image-super-resolution-app.onrender.com/docs
```

This opens the auto-generated Swagger UI where you can upload images and try the API interactively.

---

## Local Development

To run the API locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn api:app --reload
```

Then open `http://127.0.0.1:8000/docs` in your browser.

---

## API DEMO

![FastAPI - Swagger UI_page-0001](https://github.com/user-attachments/assets/4b02cc5b-57ee-4f96-8cf2-68e952fb2568)


To run the original Streamlit app instead:

```bash
streamlit run app.py
```
---

## Author
Arkaprava Roy
