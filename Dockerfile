FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for git (CLIP install)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (use CPU wheels for torch/torchvision by default)
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt && \
    pip install git+https://github.com/openai/CLIP.git

# Copy source and model
COPY app ./app
COPY models ./models

EXPOSE 8000

ENV MODEL_PATH=/app/models/best_model.pth

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
