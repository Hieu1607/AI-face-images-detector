# BTL-Deep-Learning

Binary classifier for real vs. fake face images using a frozen CLIP ViT-L/14 backbone with a lightweight linear head. Training and evaluation are driven from the notebook notebooks/train-real-vs-fake-face-classifier.ipynb.

Easily understand main content by accessing : https://www.kaggle.com/code/nguyenhieu1607/train-real-vs-fake-face-classifier

## Data
- Real: CelebA subset under data/processed/sample_1pct/celeba/ (label 0)
- Fake: three sources under data/processed/sample_1pct/ — fairfacegen/, person_face_dataset/, stable_diffusion_faces/ (label 1)
- Allowed extensions: .jpg, .jpeg, .png, .webp, .bmp

## Environment
- Python 3.10+ recommended, GPU strongly suggested
- Install deps: `pip install -r requirements.txt`
- Kaggle access: place kaggle.json in project root or your Kaggle config dir to run download notebooks

## Training (Notebook)
Open notebooks/train-real-vs-fake-face-classifier.ipynb and run top-to-bottom:
- Paths: adjust DATA_ROOT and MODEL_SAVE_PATH (defaults target Kaggle /kaggle/input/... and /kaggle/working/; set to local data/processed/sample_1pct if training locally)
- Model: CLIP ViT-L/14 frozen encoder + 3-layer MLP head with sigmoid
- Augmentation: resize→random crop 224, flip, rotation 15°, color jitter, Gaussian blur; val/test use center crop
- Splits: per-source train/val/test, then merged and shuffled
- Training: BCELoss, Adam (lr 1e-3), ReduceLROnPlateau, batch size 32, early stopping patience 5
- Outputs: best_model.pth and final_model.pth stored at MODEL_SAVE_PATH; history and metrics saved with checkpoints

## Evaluation & Visualization
- Validation/test metrics logged each epoch; test evaluation runs after loading best_model.pth
- Section "Visualize Predictions" in the notebook plots sample predictions with confidence
- Section "Plot Training History" renders loss/accuracy curves and prints best val acc and final test acc

## Deployment & API

### FastAPI Backend (app/main.py)
REST API service for real-time face classification predictions:
- **Endpoint**: `POST /predict` - Upload an image and get real/fake prediction with confidence score
- **Response**: JSON with `prediction` (0=real, 1=fake), `confidence`, `label`, and `is_fake` boolean
- **Model Loading**: Automatically loads `models/best_model.pth` on startup
- **Image Processing**: Accepts common image formats, applies CLIP preprocessing transforms
- **Auto-install**: Attempts to install CLIP package if missing at runtime

#### Running the API
See [QUICKSTART.md](QUICKSTART.md) for detailed instructions. Quick options:

**Option 1 - Docker (Recommended)**:
```bash
docker compose up --build
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Option 2 - Local Uvicorn**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Example Request**:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

### Gradio Web Interface (app/gradio_app.py)
Interactive web UI for testing the classifier:
- **Features**: Drag-and-drop image upload, real-time predictions, confidence visualization
- **Demo Images**: Includes sample images in `images_to_demo/` folder for quick testing
- **Batch Testing**: Test multiple images in gallery view
- **Visual Feedback**: Color-coded results (red for fake, green for real) with confidence percentages

**Running Gradio App**:
```bash
python app/gradio_app.py
# Opens browser at http://localhost:7860
```

### Docker Configuration
- **Dockerfile**: Optimized multi-stage build with CPU PyTorch wheels (customize for GPU)
- **docker-compose.yml**: Single-service setup with model volume mounting
- **Port**: 8000 (configurable in docker-compose.yml)
- **Restart Policy**: `unless-stopped` for production reliability
- **Volume**: Models mounted read-only from `./models` for live updates without rebuild

## Image Processing Utilities (src/process/)
Standalone preprocessing scripts for data preparation and augmentation:

### resize_images.py
- `load_image_with_opencv()`: Load images with BGR→RGB conversion
- `resize_image()`: Resize to target dimensions (default 224×224 for CLIP)

### normalize_images.py
- `standardize_normalize()`: Convert pixel values from [0, 255] to [0.0, 1.0] (float32)

### augment_images.py
Data augmentation functions for training robustness:
- `augment_flip()`: Horizontal/vertical/both flipping
- `augment_rotation()`: Rotate around center with angle control
- Additional augmentations: brightness, contrast, blur, noise, color jitter

**Usage Example**:
```python
from src.process import resize_images, normalize_images, augment_images
import cv2

# Load and preprocess
img = resize_images.load_image_with_opencv("path/to/image.jpg")
img_resized = resize_images.resize_image(img, (224, 224))
img_norm = normalize_images.standardize_normalize(img_resized)

# Augmentation
img_flipped = augment_images.augment_flip(img_resized, flip_code=1)
img_rotated = augment_images.augment_rotation(img_resized, angle=15)
```

## Data Download Utilities (src/data/)
Scripts for automated dataset acquisition:
- **download_from_kaggle.py**: Programmatic Kaggle dataset downloads (requires kaggle.json credentials)
- **download_from_gg_drive.py**: Google Drive file/folder downloads with authentication handling

## Notebooks
- **train-real-vs-fake-face-classifier.ipynb**: Main training pipeline
- **download_data.ipynb**: Automated dataset downloading from Kaggle/Drive
- **download_diffution_images.ipynb**: Fetch AI-generated faces from diffusion models
- **preprocess_data.ipynb**: Batch preprocessing, resizing, and normalization
- **sample_1pct.ipynb**: Create 1% subset for rapid prototyping

## Configuration
- **configs/logger.py**: Centralized logging setup with YAML configuration
- **configs/logger.yml**: Log levels, formats, handlers, and file outputs

## Development Notes
- Keep folder layout intact (src/, data/, models/, notebooks/, configs/, app/)
- Use feature branches and run notebook or unit checks before merging
- For API changes, update both FastAPI and Gradio apps to maintain parity
- Test Docker builds locally before pushing: `docker compose up --build`
- GPU support: Modify Dockerfile torch installation for CUDA wheels
