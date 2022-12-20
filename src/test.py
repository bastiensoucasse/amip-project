import sys

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import avg_pool2d
from torchvision.transforms.functional import gaussian_blur


def sr_test(name: str, scaling_factor: int, input: str):
    """
    Test a super resolution model.

    Parameters:
    - name (str): The name to save the trained model in the models directory.
    - scaling_factor (int): The upsampling factor of the model (should be 4 or 8).
    - input (str): The path of the image to test the model on.
    """

    # Set up the device
    if torch.__version__ < "1.12":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device.type}.")

    # Set up the data pre-processing
    pre_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Set up the data post-processing
    post_transform = transforms.Compose([
        # transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[-0.229, -0.224, -0.225]),
        transforms.ToPILImage()
    ])

    # Open the input image
    input_img = Image.open(input).convert("RGB")

    # Crop the input image to a centered sub-part of size 288x288
    width, height = input_img.size
    left = (width - 288) // 2
    top = (height - 288) // 2
    right = (width + 288) // 2
    bottom = (height + 288) // 2
    input_img = input_img.crop((left, top, right, bottom))

    # Pre-process the input image
    input_img = pre_transform(input_img)

    # Generate the low-resolution image by blurring and downsampling the high-resolution image
    input_img = gaussian_blur(input_img, 5, 1)
    input_img = avg_pool2d(input_img, kernel_size=4, stride=4)

    # Move data to the device
    input_img = input_img.to(device)

    # Load the image transformer model
    image_transformer = torch.load(f"models/{name}.pth").to(device)
    image_transformer.eval()

    # Generate the high resolution version of the low resolution input image
    generated_img = image_transformer(input_img)

    # Move back data from the device
    input_img = input_img.cpu()
    generated_img = generated_img.cpu()

    # Post-process the images
    input_img = post_transform(input_img)
    generated_img = post_transform(generated_img)

    return input_img, generated_img
