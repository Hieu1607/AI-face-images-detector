# Real vs Fake Face Classifier - Quickstart

This guide shows how to quickly get started with the Real vs Fake Face Classifier. You can use either the **FastAPI REST API** for programmatic access or the **Gradio Web UI** for interactive testing.

## Prerequisites

- **Model**: Ensure `models/best_model.pth` exists (trained using [train-real-vs-fake-face-classifier.ipynb](notebooks/train-real-vs-fake-face-classifier.ipynb))
- **Python**: 3.10+ recommended
- **Docker**: Optional, for containerized deployment

---

# Option A: Gradio Web Interface (Recommended for Testing)

The Gradio app provides an interactive web UI for quick testing and visualization.

## 1) Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

**For GPU (CUDA 11.8 example)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 2) Run Gradio App

```bash
python app/gradio_app.py
```

The app will:
- Auto-open in your browser at `http://localhost:7860`
- Load `models/best_model.pth` automatically
- Provide sample images from `images_to_demo/` folder

## 3) Using the Interface

1. **Upload Image**: Drag & drop or click to upload
2. **View Results**: 
   - Green background = Real face (confidence %)
   - Red background = Fake face (confidence %)
3. **Test Samples**: Use provided demo images for quick testing
4. **Batch Processing**: Test multiple images via gallery view

---

# Option B: FastAPI REST API (Production)

For programmatic access and integration with other services.

## 1) Run with Docker (Recommended)

**Build and Start**:
```bash
docker compose up --build
```

**Access Points**:
- API: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

**Test API**:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your_image.jpg"
```

**Example Response**:
```json
{
  "prediction": 1,
  "confidence": 0.9234,
  "label": "Fake",
  "is_fake": true
}
```

**Stop Container**:
```bashLocally with Uvicorn

**Install Dependencies**:
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

**Start Server**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Test API** (same as Docker):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your_image.jpg"
```

**Python Example**:
```python
import requests

url = "http://127.0.0.1:8000/predict"
files = {"file": open("test_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Output**:
```json
{
  "prediction": 0,
  "confidence": 0.8765,
  "label": "Real",
  "is_fake": false
}
```

---

# Advanced Usage

## Testing with Multiple Images

**Batch Script (PowerShell)**:
```powershell
Get-ChildItem "images_to_demo/*.jpg" | ForEach-Object {
    Write-Host "Testing: $($_.Name)"
    curl -X POST "http://127.0.0.1:8000/predict" `
      -H "Content-Type: multipart/form-data" `
      -F "file=@$($_.FullName)"
}
```

**Batch Script (Bash)**:
```bash
for img in images_to_demo/*.jpg; do
    echo "Testing: $img"
    curl -X POST "http://127.0.0.1:8000/predict" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@$img"
done
```

## Performance Tips

1. **GPU Acceleration**: 
   - Install CUDA-enabled PyTorch for 5-10x speedup
   - Modify Dockerfile for GPU: `FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime`

2. **Batch Processing**: 
   - For multiple images, use Gradio's batch mode or write custom batch API endpoint

3. **Model Optimization**:
   - Convert to ONNX for inference optimization
   - Use TorchScript for deployment (`torch.jit.script`)

---

# Troubleshooting

## Common Issues

**1. CLIP Import Error**:
```bash
pip install git+https://github.com/openai/CLIP.git --force-reinstall
```

**2. CUDA Out of Memory**:
- Use CPU mode: `CUDA_VISIBLE_DEVICES="" uvicorn app.main:app`
- Reduce batch size if using custom batching

**3. Model Not Found**:
```
RuntimeError: Model checkpoint not found at: models/best_model.pth
```
**Solution**: Train model first using [train-real-vs-fake-face-classifier.ipynb](notebooks/train-real-vs-fake-face-classifier.ipynb)

**4. Port Already in Use**:
```bash
# Change port in command
uvicorn app.main:app --port 8080

# Or kill existing process (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**5. Docker Build Fails**:
- Check Docker daemon is running
- Ensure sufficient disk space (2GB+ for image)
- Try: `docker system prune -a` to clear cache

## Performance Benchmarks

- **CPU**: ~200-300ms per image (Intel i7)
- **GPU**: ~20-30ms per image (NVIDIA RTX 3080)
- **Model Size**: ~1.6GB (CLIP ViT-L/14 + classifier head)

---

# Next Steps

- **Training**: See [notebooks/train-real-vs-fake-face-classifier.ipynb](notebooks/train-real-vs-fake-face-classifier.ipynb) for custom training
- **Data Preparation**: Use [notebooks/preprocess_data.ipynb](notebooks/preprocess_data.ipynb) for dataset preprocessing
- **API Integration**: Check `/docs` endpoint for full OpenAPI specification
- **Deployment**: Configure Nginx reverse proxy, SSL certificates for production

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your_image.jpg"
```

Troubleshooting:
- If CLIP fails to import, reinstall with `pip install git+https://github.com/openai/CLIP.git`.
- If torch/torchvision fail to install, use the wheel index URL that matches your CUDA version from https://download.pytorch.org/whl/torch_stable.html.
