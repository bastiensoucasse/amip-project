import sys

import torch
from torch.utils.data import DataLoader

import data_loader

BATCH_SIZE: int = 128

if __name__ == '__main__':
    # Retrieve name.
    name: str = sys.argv[1] if len(sys.argv) > 1 else 'default'
    print(f"Name: {name}.")

    # Set up device.
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device.type}.")

    # Retrieve data loaders.
    set5: DataLoader = data_loader.load('set5', batch_size=BATCH_SIZE)
    set14: DataLoader = data_loader.load('set14', batch_size=BATCH_SIZE)
    bsd100: DataLoader = data_loader.load('bsd100', batch_size=BATCH_SIZE)
