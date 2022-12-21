import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image


def display_image(image: Image.Image) -> None:
    """Displays an image."""

    # Display the image in the plot
    plt.imshow(image)
    plt.axis("off")

    # Show the figure
    plt.show()


def display_tensors(tensor: torch.Tensor) -> None:
    """Displays an image tensor."""

    # Convert tensor to image
    image = transforms.ToPILImage()(tensor)

    # Call the display image utility
    display_image(image)


def display_images(images: "list[Image.Image]") -> None:
    """Displays a list of images side to side."""

    if len(images) == 0:
        return

    if len(images) == 1:
        display_image(images[0])
        return

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
