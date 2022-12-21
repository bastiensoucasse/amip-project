import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets import SuperResolutionDataset
from models import ImageTransformer, SuperResolutionLoss


def sr_train(name: str, scaling_factor: int, use_pixel_loss: bool = False, num_epochs: int = 16, batch_size: int = 32) -> None:
    """
    Train a super resolution model.

    Parameters:
    - name (str): The name to save the trained model in the models directory.
    - scaling_factor (int): The upsampling factor of the model (should be 4 or 8).
    - use_pixel_loss (bool, optional): True if the super resolution loss should use the pixel loss improvement.
    - num_epochs (int, optional): The number of epochs to train the model for.
    - batch_size (int, optional): The batch size to use during training.
    """

    # Set up the device
    if torch.__version__ < "1.12":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device.type}.")

    # Set up the data pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the custom super resolution dataset and apply the pre-prcessing
    sr_dataset = SuperResolutionDataset("data", scaling_factor, transform=transform)

    # Define the custom super resolution data loader
    sr_data = DataLoader(sr_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create an instance of the image transformer model
    image_transformer = ImageTransformer(scaling_factor).to(device)

    # Define the optimizer and learning rate
    optimizer = torch.optim.Adam(image_transformer.parameters(), lr=1e-3)

    # Define the super resolution loss function
    super_resolution_loss = SuperResolutionLoss(use_pixel_loss=use_pixel_loss).to(device)

    # Iterate over the number of epochs
    for epoch in range(num_epochs):
        # Print the epoch and start the epoch
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_time = time.time()

        # Iterate over the training data
        for lr_img, hr_img in sr_data:
            # Move data to the device
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Generate the high resolution version of the low resolution input image
            gen_img = image_transformer(lr_img)

            # Compute the loss
            loss = super_resolution_loss(gen_img, hr_img)

            # Backpropagate the loss and update the model parameters
            loss.backward()
            optimizer.step()

        # Compute the epoch time
        epoch_time = time.time() - epoch_time

        # Print the epoch summary
        print(f"{epoch_time:.2f}s - loss: {loss.item():.4f}")

    # Save the trained model
    models_dir = "models"
    Path(f"{models_dir}").mkdir(parents=True, exist_ok=True)
    torch.save(image_transformer.state_dict(), f"{models_dir}/{name}.pth")
    print(f"Super resolution model \"{name}\" saved\".")


if __name__ == "__main__":
    # Check if the correct number of command line arguments are provided
    if len(sys.argv) != 6:
        print(f"Train a super resolution model.")
        print(f"Usage: {sys.argv[0]} <name> <scaling_factor> <use_pixel_loss> <num_epochs> <batch_size>")
        exit(-1)

    # Parse command line arguments
    name: str = sys.argv[1]
    scaling_factor: int = int(sys.argv[2])
    use_pixel_loss: bool = sys.argv[3].lower() in ['true', '1', 't', 'y', 'yes']
    num_epochs: int = int(sys.argv[4])
    batch_size: int = int(sys.argv[5])

    # Train a super resolution model
    sr_train(name, scaling_factor,
             use_pixel_loss=use_pixel_loss,
             num_epochs=num_epochs,
             batch_size=batch_size)
