# train.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import CNN_to_RNN
from data_preprocessing import train_loader, val_loader, vocab
import wandb
import json
from tqdm import tqdm

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In your model or training script
vocab_size = len(vocab)

def get_serializable_config(config):
    serializable_config = {}
    for key, value in vars(config).items():
        try:
            json.dumps(value)  # Try to serialize the value to JSON
            serializable_config[key] = value  # If no error is raised, add the item to the dictionary
        except TypeError:
            pass  # If a TypeError is raised, ignore the item
    return serializable_config

def train(config=None):
    # Initialize wandb
    run = wandb.init()
    config = run.config


    # Initialize the model, loss function, and optimizer
    model = CNN_to_RNN(config.embed_size, config.hidden_size, vocab_size, config.num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Initialize the minimum validation loss with infinity
    min_val_loss = float('inf')

    # Initialize the patience counter
    patience_counter = 0

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0  # Initialize total_train_loss
        total_train_samples = 0  # Initialize total_train_samples
        for i, (images, captions, lengths) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            # Move data to the correct device
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward pass
            outputs = model(images, captions)
            outputs = outputs.view(-1, vocab_size)  # Reshape outputs to [batch_size * seq_len, vocab_size]
            outputs = outputs[:targets.size(0)]  # Remove the extra outputs due to padding
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Update total_train_loss and total_train_samples
            total_train_loss += loss.item() * images.size(0)
            total_train_samples += targets.size(0)

        # Calculate and log the average training loss
        avg_train_loss = total_train_loss / total_train_samples
        wandb.log({"Training Loss": avg_train_loss,})

        # Validation phase
        model.eval()
        with torch.no_grad():
            total_loss = 0  # Initialize total_loss
            total_samples = 0  # Initialize total_samples
            total_correct = 0  # Initialize total_correct
            for i, (images, captions, lengths) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                outputs = model(images, captions)
                outputs = outputs.view(-1, vocab_size)  # Reshape outputs to [batch_size * seq_len, vocab_size]
                outputs = outputs[:targets.size(0)]  # Remove the extra outputs due to padding
                loss = criterion(outputs, targets)
                total_loss += loss.item() * images.size(0)
                total_samples += targets.size(0)

                # Calculate the number of correct predictions
                _, predicted = outputs.max(1)
                total_correct += (predicted == targets).sum().item()

            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            print(f"Validation Loss: {avg_loss}, Accuracy: {accuracy}")

            # Log the validation loss, accuracy, and hyperparameters to wandb
            wandb.log({"Validation Loss": avg_loss,"Accuracy": accuracy,})


            # Save the model if the validation loss decreased
            if avg_loss < min_val_loss:
                print(f"Validation loss decreased ({min_val_loss} --> {avg_loss}). Saving model ...")
                torch.save(model.state_dict(), 'model.pt')
                min_val_loss = avg_loss
                patience_counter = 0  # reset the patience counter
            else:
                patience_counter += 1

            # If the validation loss hasn't improved for `patience` epochs, stop training
            if patience_counter >= config.patience:
                print("Early stopping")
                break

            # Update the learning rate
            scheduler.step(avg_loss)
            
    # Save the config object to a file after training
    with open('config.json', 'w') as f:
        serializable_config = get_serializable_config(config)
        json.dump(serializable_config, f)

    return config