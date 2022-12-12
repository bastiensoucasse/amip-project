import sys

import torch
from torchvision import models

import data_loader
from models import ImageTransformNet

if __name__ == "__main__":
    # Retrieve name.
    name = sys.argv[1] if len(sys.argv) > 1 else "default"
    print(f"Name: {name}.")

    # Set up device.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device.type}.")

    # Retrieve data loaders.
    set5 = data_loader.load("set5", batch_size=4)
    set14 = data_loader.load("set14", batch_size=4)
    bsd100 = data_loader.load("bsd100", batch_size=4)

    # Define the image transformer.
    image_transformer = ImageTransformNet().to(device)
    image_transformer_optimizer = torch.optim.Adam(image_transformer.parameters(), lr=1e-3)

    # Define the VGG-16 model.
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg16_optimizer = torch.optim.Adam(vgg16.parameters(), lr=1e-3)
