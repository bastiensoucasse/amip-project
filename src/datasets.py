import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import gaussian_blur


class SuperResolutionDataset(torchvision.datasets.VisionDataset):
    """Custom dataset for generating low-resolution and high-resolution image pairs for super resolution tasks.

    Args:
        root_dir (str): Root directory of the dataset.
        scaling_factor (int): Factor of downsampling (should be 4 or 8).
        transform (callable, optional): Optional transform to be applied to the high-resolution images.
    """

    def __init__(self, root_dir: str, scaling_factor: int, transform: Optional[Callable] = None) -> None:
        # Save parameters
        self.root_dir = root_dir
        self.scaling_factor = scaling_factor
        self.transform = transform

        # Load the list of image file names
        self.images = self.load_data()

    def __len__(self) -> int:
        """Returns the length of the dataset, which is equal to the number of image file names."""

        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates a low-resolution and high-resolution image pair for the given index.

        Args:
            idx (int): Index of the image pair to generate.

        Returns:
            tuple: Tuple containing the low-resolution image tensor and the high-resolution image tensor.
        """

        # Load the image at the given index
        img_path = os.path.join(self.root_dir, self.images[idx])
        hr_img = Image.open(img_path).convert("RGB")

        # Crop the image to a centered sub-part of size 288x288
        width, height = hr_img.size
        left = (width - 288) // 2
        top = (height - 288) // 2
        right = (width + 288) // 2
        bottom = (height + 288) // 2
        hr_img = hr_img.crop((left, top, right, bottom))

        # Apply the transform to the high-resolution image, if specified
        if self.transform is not None:
            hr_img = self.transform(hr_img)
        else:
            hr_img = transforms.ToTensor()(hr_img)

        # Generate the low-resolution image by blurring and downsampling the high-resolution image
        lr_img = hr_img.clone()
        lr_img = gaussian_blur(lr_img, 3, 1)
        lr_img = F.avg_pool2d(lr_img, kernel_size=self.scaling_factor, stride=self.scaling_factor)

        # Return the low-resolution and high-resolution images
        return lr_img, hr_img

    def load_data(self) -> list[str]:
        """Loads the list of image file names from the root directory."""

        # Load the list of image file names
        images = []
        for file in os.listdir(self.root_dir):
            # Handle unwanted files
            if file in ['.DS_Store']:
                break

            images.append(file)
        return images
