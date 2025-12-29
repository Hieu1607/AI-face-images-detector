import io
import os
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Optional runtime install of CLIP if missing, similar to notebook behavior
try:
    import clip  # type: ignore
except Exception:
    try:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/openai/CLIP.git'])
        import clip  # type: ignore
    except Exception as e:
        clip = None  # We will raise a clear error at startup


app = FastAPI(title="Real vs Fake Face Classifier API")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'best_model.pth'
IMG_SIZE = 224


class CLIPClassifier(nn.Module):
    """CLIP visual backbone (frozen) + small linear head for binary classification."""
    def __init__(self, clip_model, freeze_backbone: bool = True):
        super().__init__()
        self.clip_visual = clip_model.visual
        self.clip_visual.eval()

        if freeze_backbone:
            for p in self.clip_visual.parameters():
                p.requires_grad = False

        clip_dim = 768  # CLIP ViT-L/14 visual output dim
        self.head = nn.Sequential(
            nn.Linear(clip_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match dtype expected by CLIP visual encoder
        x = x.to(self.clip_visual.conv1.weight.dtype)
        with torch.no_grad():
            features = self.clip_visual(x)
        out = self.head(features.float())  # [B, 1]
        return out.view(-1)  # [B]


def build_preprocess() -> transforms.Compose:
    """Validation-style preprocessing used during training in the notebook."""
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])


@app.on_event('startup')
async def startup_event():
    if clip is None:
        raise RuntimeError("CLIP library is not available and auto-install failed.")

    # Load CLIP model
    clip_model, _ = clip.load("ViT-L/14", device=DEVICE)

    # Init classifier and load weights
    model = CLIPClassifier(clip_model, freeze_backbone=True).to(DEVICE)

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model checkpoint not found at: {MODEL_PATH}")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")

    # Attach to app state
    app.state.model = model
    app.state.preprocess = build_preprocess()


@app.post('/predict')
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Predict whether the uploaded image is Real (0) or Fake (1).
    Returns predicted class, confidence, and class probabilities.
    """
    # Validate content type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Preprocess
    preprocess = getattr(app.state, 'preprocess', None)
    model = getattr(app.state, 'model', None)
    if preprocess is None or model is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

    with torch.no_grad():
        output = model(tensor)  # [1]
        prob_fake = float(output.item())
        prob_real = 1.0 - prob_fake
        pred_label = 1 if prob_fake > 0.5 else 0
        predicted_class = 'Fake' if pred_label == 1 else 'Real'
        confidence = max(prob_fake, prob_real)

    return JSONResponse({
        'predicted_label': pred_label,
        'predicted_class': predicted_class,
        'confidence': round(confidence, 6),
        'probabilities': {
            'Real': round(prob_real, 6),
            'Fake': round(prob_fake, 6)
        }
    })
