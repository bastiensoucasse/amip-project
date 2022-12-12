import matplotlib.pyplot as plt
import torch
from PIL import Image as PIL
from torchvision import transforms


def image_to_tensor(image: PIL.Image, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Converts an image into a tensor (onto a device if specified)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]),
        transforms.Lambda(lambda x: x.mul_(255)),
    ])

    return transform(image).unsqueeze(0).to(device)


def tensor_to_image(tensor: torch.Tensor) -> PIL.Image:
    """Converts a tensor into an image."""
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1. / 255.)),
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1, 1, 1]),
        transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])])
    ])

    tmp = transform(tensor.squeeze(0))
    tmp[tmp > 1] = 1
    tmp[tmp < 0] = 0

    transform = transforms.Compose([
        transforms.ToPILImage()
    ])

    return transform(tmp)


def display_image(image: PIL.Image, size: tuple[int, int] = (4, 4)) -> None:
    """Visualizes an image."""
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def display_images(images: list[PIL.Image], size: tuple[int, int] = (8, 4)) -> None:
    """Visualizes an image list."""
    fig = plt.figure(figsize=size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()


def gram_matrix(input: torch.Tensor) -> torch.Tensor:
    """Computes a Gram matrix."""
    b, d, h, w = input.size()
    features = input.view(d, h * w)
    gramm = torch.mm(features, features.t())
    return gramm.div(b * d * h * w)
