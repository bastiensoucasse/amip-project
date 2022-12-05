import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MODEL = 'MyResNetDA'

CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = len(CLASSES)
NUM_CHANNELS = 3

BATCH_SIZE = 32
NUM_EPOCHS = 20

PLOT = True


if __name__ == '__main__':
    # Check custom model.
    if len(sys.argv) > 1:
        MODEL = sys.argv[1]

    # Set up device.
    if torch.__version__ < '1.12':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}.")

    # Load and augment the data.
    train_transform = transforms.Compose([
        transforms.RandomAffine(0, scale=(.8, 1.2), shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.CIFAR10('data', train=True, transform=train_transform, download=False)
    test_dataset = datasets.CIFAR10('data', train=False, transform=test_transform, download=False)
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Define the model.
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Change the classifier section.
    model.fc = nn.LazyLinear(NUM_CLASSES)

    # Send model to device.
    model = model.to(device)
    print(f"Model: {MODEL}.")

    # Define optimizer and criterion.
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train the model.
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    training_start_time = time.time()
    for e in range(NUM_EPOCHS):
        print(f'Epoch {e + 1}/{NUM_EPOCHS}')
        epoch_start_time = time.time()

        # training
        model.train()
        running_loss, running_accuracy = 0, 0
        for x, y in train_data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / len(train_data_loader)
            running_accuracy += (pred.argmax(1) == y).sum().item() / BATCH_SIZE / len(train_data_loader)

        # validation
        with torch.no_grad():
            model.eval()
            running_val_loss, running_val_accuracy = 0, 0
            for x, y in test_data_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                running_val_loss += loss.item() / len(test_data_loader)
                running_val_accuracy += (pred.argmax(1) == y).sum().item() / BATCH_SIZE / len(test_data_loader)

        epoch_time = time.time() - epoch_start_time
        history['loss'] += [running_loss]
        history['accuracy'] += [running_accuracy]
        history['val_loss'] += [running_val_loss]
        history['val_accuracy'] += [running_val_accuracy]
        print(f'{epoch_time:.2f}s - loss: {running_loss:.4f} - accuracy: {running_accuracy:.4f} - val_loss: {running_val_loss:.4f} - val_accuracy: {running_val_accuracy:.4f}')
    training_time = time.time() - training_start_time

    # Evaluate the model.
    with torch.no_grad():
        y_test, y_pred = [], []
        evaluating_start_time = time.time()
        model.eval()
        running_loss, running_accuracy = 0, 0
        for x, y in test_data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            running_loss += loss.item() / len(test_data_loader)
            running_accuracy += (pred.argmax(1) == y).sum().item() / BATCH_SIZE / len(test_data_loader)

            # Store the expected values and the predictions, for the confusion matrix and to call ten_worst.
            y_test.extend(y.detach().cpu().numpy())
            y_pred.extend(pred.detach().cpu().numpy())
        evaluating_time = time.time() - evaluating_start_time
        y_test, y_pred = np.array(y_test), np.array(y_pred)
        print(f'{evaluating_time:.2f}s - loss: {running_loss:.4f} - accuracy: {running_accuracy:.4f}')

    # Display the summary.
    print(f'SUMMARY:\n    - Loss: {running_loss:.4f}\n    - Accuracy: {running_accuracy:.4f}\n    - Training Time: {training_time:.2f}s')

    if not PLOT:
        exit()

    # Create folder if necessary
    Path(f'plots/{MODEL}/').mkdir(parents=True, exist_ok=True)

    # Plot the loss.
    plt.plot(history['loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epoch')
    plt.savefig(f'plots/{MODEL}/loss.png')
    plt.clf()

    # Plot the accuracy.
    plt.plot(history['accuracy'], label='Training')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epoch')
    plt.savefig(f'plots/{MODEL}/accuracy.png')
    plt.clf()

    # Plot the confusion matrix.
    cm = confusion_matrix(y_test, y_pred.argmax(1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap=plt.cm.magma)
    plt.savefig(f'plots/{MODEL}/confusion_matrix.png')
    plt.clf()
