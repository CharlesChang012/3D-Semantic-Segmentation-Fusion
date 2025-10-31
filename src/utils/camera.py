from transformers import AutoModel, AutoImageProcessor
import torch
from PIL import Image


class ImageFeatureEncoder:
    def __init__(self, model_name="dinov3", device="cuda"):
        self.device = device
        self.model_name = model_name.lower()
        print("===========================================================")
        print(f"[INFO] Initializing ImageFeatureEncoder: {self.model_name}")
        print(f"[INFO] Using device: {self.device}")
        print("===========================================================")
        
        if self.model_name == "dinov2":
            model_id = "facebook/dinov2-small"
            self.patch_size = 14
        elif self.model_name == "dinov3":
            model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
            self.patch_size = 16
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # Preprocessor and model
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.resize_size = self.processor.size["height"]
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()


    def __call__(self, images):
        patch_features, global_features = [], []

        with torch.inference_mode(), torch.autocast(
            device_type=self.device,
            dtype=torch.float16,
        ):
            for img in images:
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                feats = outputs.last_hidden_state.squeeze(0)
                cls_token = feats[0]
                patch_tokens = feats[1:-4] # shape: (M, 384) exclude register tokens
                patch_features.append(patch_tokens.cpu())
                global_features.append(cls_token.cpu())

        return {
            "patch_features": torch.stack(patch_features),
            "global_features": torch.stack(global_features),
        }
