import os
import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Optional runtime install of CLIP if missing
try:
    import clip  # type: ignore
except Exception:
    try:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/openai/CLIP.git'])
        import clip  # type: ignore
    except Exception as e:
        clip = None


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'best_model.pth'
DEMO_IMAGES_PATH = Path(__file__).resolve().parents[1] / 'images_to_demo'
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


def load_model():
    """Load the trained model."""
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

    return model


# Initialize model and preprocessing
print(f"Loading model from {MODEL_PATH}...")
print(f"Using device: {DEVICE}")
model = load_model()
preprocess = build_preprocess()
print("Model loaded successfully!")


def predict_image(image):
    """
    Predict whether the uploaded image is Real or Fake.
    
    Args:
        image: PIL Image or numpy array from Gradio
        
    Returns:
        tuple: (predicted_class, confidence_dict)
    """
    if image is None:
        return "Please upload an image", {}
    
    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Ensure RGB mode
        image = image.convert('RGB')
        
        # Preprocess
        tensor = preprocess(image).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
        
        # Predict
        with torch.no_grad():
            output = model(tensor)  # [1]
            prob_fake = float(output.item())
            prob_real = 1.0 - prob_fake
            pred_label = 1 if prob_fake > 0.5 else 0
            predicted_class = '🤖 FAKE (AI-Generated)' if pred_label == 1 else '✅ REAL (Authentic)'
            confidence = max(prob_fake, prob_real) * 100
        
        # Prepare confidence dictionary for label output
        confidence_dict = {
            'Real (Authentic)': prob_real,
            'Fake (AI-Generated)': prob_fake
        }
        
        result_text = f"{predicted_class}\nConfidence: {confidence:.2f}%"
        
        return result_text, confidence_dict
        
    except Exception as e:
        return f"Error processing image: {str(e)}", {}


# Get demo images
demo_images = []
if DEMO_IMAGES_PATH.exists():
    for img_file in DEMO_IMAGES_PATH.glob("*.jpg"):
        demo_images.append(str(img_file))
    for img_file in DEMO_IMAGES_PATH.glob("*.png"):
        demo_images.append(str(img_file))
    demo_images.sort()


# Create Gradio interface
with gr.Blocks(title="Real vs Fake Face Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 Real vs Fake Face Classifier
        ### Deep Learning Model - CLIP-based Binary Classification
        
        Upload an image or select from examples to detect whether a face is **real** or **AI-generated/fake**.
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Image",
                type="pil",
                height=400
            )
            
            predict_btn = gr.Button("🔎 Analyze Image", variant="primary", size="lg")
            
            gr.Markdown("### 📸 Demo Images")
            gr.Examples(
                examples=demo_images,
                inputs=input_image,
                label="Click to load example"
            )
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Prediction Result",
                lines=3,
                interactive=False
            )
            
            output_label = gr.Label(
                label="Confidence Scores",
                num_top_classes=2
            )
            
            gr.Markdown(
                """
                ### 📊 Model Information
                - **Architecture**: CLIP ViT-L/14 + Custom Classification Head
                - **Input Size**: 224x224 pixels
                - **Classes**: Real (0) | Fake (1)
                - **Device**: {}
                
                ### ℹ️ How to Use
                1. Upload an image or select from examples
                2. Click "Analyze Image" button
                3. View the prediction and confidence scores
                """.format(DEVICE)
            )
    
    # Event handlers
    predict_btn.click(
        fn=predict_image,
        inputs=input_image,
        outputs=[output_text, output_label]
    )
    
    # Auto-predict when image is uploaded
    input_image.change(
        fn=predict_image,
        inputs=input_image,
        outputs=[output_text, output_label]
    )
    
    gr.Markdown(
        """
        ---
        <div style="text-align: center;">
            <p>Powered by PyTorch & CLIP | Built with Gradio</p>
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
