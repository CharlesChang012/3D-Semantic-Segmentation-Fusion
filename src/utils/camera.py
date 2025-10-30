from transformers import pipeline
import torch
from huggingface_hub import login
import torchvision
from torchvision.transforms import v2
import torch.nn.functional as F
from torchvision import transforms
import os

login(token="YOUR_HUGGING_FAECE_TOKEN_HERE")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DINOV3_PATH = os.path.join(REPO_ROOT, 'dinov3')
DINOV3_WEIGHTS = os.path.join(DINOV3_PATH, 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')

def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

class ImageFeatureEncoder:
    def __init__(self, model_name="dinov3"):
        # Load local DINOv3 model
        if model_name == "dinov2":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.resize_size = 224
            self.patch_size = 14
        else:
            self.model = torch.hub.load(DINOV3_PATH, 'dinov3_vits16', source='local', weights=DINOV3_WEIGHTS)
            self.resize_size = 256
            self.patch_size = 16
        self.model.eval()
        self.transform = make_transform(resize_size=self.resize_size)

    def encode(self, images):
        features = []
        with torch.inference_mode():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for img in images:
                    x = self.transform(img).unsqueeze(0)
                    patch_tokens = self.model.get_intermediate_layers(x, n=1)[0]  # Get one layer n=1, access with index 0
                    output = patch_tokens.squeeze(0)  # (M, 384)
                    features.append(output)

        return torch.stack(features, dim=0) # (6, M, 384)

    def get_resize_size(self):
        return self.resize_size

    def get_patch_size(self):
        return self.patch_size
