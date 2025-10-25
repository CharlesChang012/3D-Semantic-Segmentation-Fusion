from transformers import pipeline
import torch
from huggingface_hub import login
import torchvision
from torchvision.transforms import v2
import torch.nn.functional as F
from torchvision import transforms

login(token="YOUR_HUGGING_FAECE_TOKEN_HERE")

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
        else:
            self.model = torch.hub.load("/home/pohsun/dinov3", 'dinov3_vits16', source='local', weights="/home/pohsun/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
            self.resize_size = 256
        self.model.eval()
        self.transform = make_transform(resize_size=self.resize_size)

    def __call__(self, images):
        features = []
        with torch.inference_mode():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for img in images:
                    x = self.transform(img).unsqueeze(0)
                    patch_tokens = self.model.get_intermediate_layers(x, n=1)[0]  # Get one layer n=1, access with index 0
                    output = patch_tokens.squeeze(0)  # (M, 384)
                    features.append(output)

        return torch.stack(features, dim=0) # (6, M, 384)
