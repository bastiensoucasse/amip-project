import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets import SuperResolutionDataset
from environment import BATCH_SIZE, DATA_DIR, MODELS_DIR
from models import ImageTransformer, SuperResolutionLoss


def sr_train(name: str, scaling_factor: int, use_pixel_loss: bool = False, num_epochs: int = 16) -> None:
    """
    Train a super resolution model.

    Parameters:
    - name (str): The name to save the trained model in the models directory.
    - scaling_factor (int): The upsampling factor of the model (should be 4 or 8).
    - use_pixel_loss (bool, optional): True if the super resolution loss should use the pixel loss improvement.
    - num_epochs (int, optional): The number of epochs to train the model for.
    """

    # Set up the device
    if torch.__version__ < "1.12":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device.type}")

    # Set up the data pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the custom super resolution dataset and apply the pre-prcessing
    sr_dataset = SuperResolutionDataset(DATA_DIR, scaling_factor, transform=transform)
    # sr_dataset = SuperResolutionDataset(DATA_DIR, scaling_factor)


    # Define the custom super resolution data loader
    sr_data = DataLoader(sr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Create an instance of the image transformer model
    image_transformer = ImageTransformer(scaling_factor).to(device)

    # Define the optimizer and learning rate
    optimizer = torch.optim.Adam(image_transformer.parameters(), lr=1e-3)

    # Define the super resolution loss function
    super_resolution_loss = SuperResolutionLoss(use_pixel_loss=use_pixel_loss).to(device)

    # Start the training
    training_time = time.time()

    best_loss = 1e6

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

        if best_loss > loss.item():
            Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(image_transformer.state_dict(), f"{MODELS_DIR}/{name}.pth")
            best_loss = loss.item()


    # Compute the training time
    training_time = time.time() - training_time

    # Print the training summary
    print(f"Trainig done in {training_time:.2f}s.")

    # Save the trained model

    print(f"Super resolution model \"{name}\" saved with best_loss: {best_loss}")


if __name__ == "__main__":
    # Check if the correct number of command line arguments are provided
    if len(sys.argv) != 5:
        print(f"Train a super resolution model")
        print(f"Usage: {sys.argv[0]} <name> <scaling_factor> <use_pixel_loss> <num_epochs>")
        exit(-1)

    # Parse command line arguments
    name: str = sys.argv[1]
    scaling_factor: int = int(sys.argv[2])
    use_pixel_loss: bool = sys.argv[3].lower() in ["true", "1", "t", "y", "yes"]
    num_epochs: int = int(sys.argv[4])

    # Train a super resolution model
    sr_train(name, scaling_factor,
             use_pixel_loss=use_pixel_loss,
             num_epochs=num_epochs)
