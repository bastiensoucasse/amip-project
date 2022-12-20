import os
from typing import Callable, Optional

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import gaussian_blur


class SuperResolutionDataset(torchvision.datasets.VisionDataset):
    """Custom dataset for generating low-resolution and high-resolution image pairs for super resolution tasks.

    Args:
        root_dir (str): Root directory of the dataset.
        transform (callable, optional): Optional transform to be applied to the high-resolution images.
        scaling_factor (int, optional): Factor of downsampling (should be 4 or 8). 
    """

    def __init__(self, root_dir: str, transform: Optional[Callable] = None, scaling_factor: int = 4):
        self.root_dir = root_dir
        self.transform = transform
        self.kernel_size = 3
        self.sigma = 1
        self.pool_size = scaling_factor
        self.stride = scaling_factor

        # Load the list of image file names
        self.images = self.load_data()

    def __len__(self):
        """Returns the length of the dataset, which is equal to the number of image file names."""

        return len(self.images)

    def __getitem__(self, idx):
        """Generates a low-resolution and high-resolution image pair for the given index.

        Args:
            idx (int): Index of the image pair to generate.

        Returns:
            tuple: Tuple containing the low-resolution image tensor and the high-resolution image tensor.
        """

        # Load the image at the given index
        img_path = os.path.join(self.root_dir, self.images[idx])
        high_res_img = Image.open(img_path).convert("RGB")

        # Crop the image to a centered sub-part of size 288x288
        width, height = high_res_img.size
        left = (width - 288) // 2
        top = (height - 288) // 2
        right = (width + 288) // 2
        bottom = (height + 288) // 2
        high_res_img = high_res_img.crop((left, top, right, bottom))

        # Apply the transform to the high-resolution image, if specified
        if self.transform is not None:
            high_res_img = self.transform(high_res_img)
        else:
            high_res_img = transforms.ToTensor()(high_res_img)

        # Generate the low-resolution image by blurring and downsampling the high-resolution image
        low_res_img = high_res_img.clone()
        low_res_img = gaussian_blur(low_res_img, self.kernel_size, self.sigma)
        low_res_img = F.avg_pool2d(low_res_img, kernel_size=self.pool_size, stride=self.stride)

        # Return the low-resolution and high-resolution images
        return low_res_img, high_res_img

    def load_data(self):
        """Loads the list of image file names from the root directory."""

        # Load the list of image file names
        images = []
        for file in os.listdir(self.root_dir):
            images.append(file)
        return images
