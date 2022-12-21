import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import avg_pool2d
from torchvision.transforms.functional import gaussian_blur

from models import ImageTransformer


def sr_test(name: str, scaling_factor: int, input: str) -> tuple[Image.Image, Image.Image]:
    """
    Test a super resolution model.

    Parameters:
    - name (str): The name to save the trained model in the models directory.
    - scaling_factor (int): The upsampling factor of the model (should be 4 or 8).
    - input (str): The path of the image to test the model on.
    """

    with torch.no_grad():
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

        # Load the image transformer model
        image_transformer = ImageTransformer(scaling_factor).to(device)
        image_transformer.load_state_dict(torch.load(f"models/{name}.pth"))
        image_transformer.eval()

        # Open the input image
        lr_img = Image.open(input).convert("RGB")

        # Pre-process the input image
        lr_img = pre_transform(lr_img)

        # Generate the low-resolution image by blurring and downsampling the high-resolution image
        lr_img = gaussian_blur(lr_img, 3, 1)
        lr_img = avg_pool2d(lr_img, kernel_size=scaling_factor, stride=scaling_factor)

        # Move data to the device
        lr_img = lr_img.unsqueeze(0).to(device)

        # Generate the high resolution version of the low resolution input image
        gen_img = image_transformer(lr_img)

        # Move back data from the device
        lr_img = lr_img.squeeze(0).cpu()
        gen_img = gen_img.squeeze(0).cpu()

        # Post-process the images
        lr_img = post_transform(lr_img)
        gen_img = post_transform(gen_img)

        return lr_img, gen_img


if __name__ == "__main__":
    # Check if the correct number of command line arguments are provided
    if len(sys.argv) != 4:
        print(f"Test a super resolution model.")
        print(f"Usage: {sys.argv[0]} <name> <scaling_factor> <input>")
        exit(-1)

    # Parse command line arguments
    name: str = sys.argv[1]
    scaling_factor: int = int(sys.argv[2])
    input: str = sys.argv[3]

    # Test the super resolution model
    lr_img, gen_img = sr_test(name, scaling_factor, input)

    # Save the images
    output_dir = "output"
    output_dir = f"{output_dir}/{name}/{Path(input).stem}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    lr_img.save(f"{output_dir}/lr_img.jpg")
    gen_img.save(f"{output_dir}/gen_img.jpg")
    print(f"Low resolution and generated images for \"{input}\" saved\".")
