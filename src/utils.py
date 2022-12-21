import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image


def display_images(images: "list[Image.Image]") -> None:
    """Displays a list of images side to side."""

    # Create a figure with two subplots
    _, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))

    # Display the images in the subplots
    for i in range(len(images)):
        axes[i].imshow(images[i])
        axes[i].axis("off")

    # Show the figure
    plt.show()


def display_tensors(tensors: "list[torch.Tensor]") -> None:
    """Displays a list of image tensors side to side."""

    # Convert tensors to images
    images = [transforms.ToPILImage()(tensor) for tensor in tensors]

    # Call the display images utility
    display_images(images)
