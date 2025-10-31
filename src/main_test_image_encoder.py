import argparse
import os
from utils.camera_test import ImageFeatureEncoder
import torch
import torchvision


def main():
    from transformers.image_utils import load_image

    # Load test image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = load_image(url)
    images_list = [image] * 6  # batch of 6 images


    #### DinoV3
    # Initialize image feature encoder
    image_encoder = ImageFeatureEncoder(model_name="dinov3")
    # Encode image features
    image_patch_tokens = image_encoder(images_list)
    print("patch feature:", image_patch_tokens["patch_features"].shape)
    print("global features:", image_patch_tokens["global_features"].shape)

if __name__ == "__main__":
    main()