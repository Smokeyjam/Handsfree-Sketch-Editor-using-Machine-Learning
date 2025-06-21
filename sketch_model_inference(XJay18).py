import torch
from transformers import AutoModelForImageClassification
from drawing import strokes_to_image
import os

class SketchClassifier:
    def __init__(self, model_dir, device='cpu'):
        self.device = torch.device(device)

        # Check if local folder
        if os.path.exists(model_dir):
            print(f"[INFO] Loading local model from {model_dir}")
            self.model = AutoModelForImageClassification.from_pretrained(
                model_dir,
                torch_dtype=torch.float32,  # Safer dtype
                low_cpu_mem_usage=True,     # Optional optimization
                #from_safetensors=False      # Important for normal PyTorch .bin files
                #ignore_mismatched_sizes=True
            ).to(self.device)
        else:
            raise ValueError(f"[ERROR] Model path {model_dir} not found!")

        self.model.eval()

    def preprocess_strokes(self, strokes):
        img = strokes_to_image(strokes, canvas_size=224)  # (224, 224)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1,224,224]
        img_tensor = img_tensor.repeat(1, 3, 1, 1)  # [1,3,224,224]

        # Manual ImageNet normalization
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - imagenet_mean) / imagenet_std

        return img_tensor

    def predict(self, strokes):
        img_tensor = self.preprocess_strokes(strokes)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=img_tensor)  # ✅ Must pass as pixel_values
            preds = outputs.logits.argmax(dim=1).item()

        return preds
