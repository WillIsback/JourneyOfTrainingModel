# evaluate.py
import torch

def evaluate(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation
        for images, captions, lengths in data_loader:
            # Move the images and captions to the current device
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            outputs = model(images, captions, lengths)

            # Calculate the loss
            loss = criterion(outputs, captions)

            # Calculate the accuracy
            accuracy = (outputs.argmax(dim=1) == captions).float().mean()

            # Update the total loss and accuracy
            total_loss += loss.item() * images.size(0)
            total_accuracy += accuracy.item() * images.size(0)
            total_samples += images.size(0)

    # Calculate the average loss and accuracy
    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples

    return avg_loss, avg_accuracy