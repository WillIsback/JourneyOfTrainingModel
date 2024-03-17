from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import os

def evaluate(model, image_folder, device):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    num_workers = int(os.cpu_count() or 1)

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    # Load the images from the image folder
    image_dataset = ImageFolder(image_folder, transform=transform)

    # Create a DataLoader for the image dataset
    test_loader = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

    model.eval()  # Set the model to evaluation mode
    generated_captions = []

    with torch.no_grad():  # Disable gradient calculation
        for images, _ in test_loader:  # Ignore image labels
            # Move the images to the current device
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Convert the outputs to captions
            captions = outputs.argmax(dim=2)  # Assuming the output shape is [batch_size, seq_len, vocab_size]
            generated_captions.extend(captions.tolist())

    return generated_captions

