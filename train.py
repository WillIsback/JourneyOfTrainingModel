# train.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import CNN_to_RNN
from data_preprocessing import train_loader, val_loader, vocab, subset_loader, Log_gpu_memory_usage
import wandb
import json
from tqdm import tqdm
import Utils.Utils as Utils


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In your model or training script
vocab_size = len(vocab)

def train(config=None):
    # If config is an instance of Config, print the config and initialize wandb
    if isinstance(config, Utils.Config):
        print("Configuration:")
        for key, value in config.__dict__.items():
            print(f"{key}: {value}")
        run = wandb.init(config=config.__dict__)
        config = run.config
        is_Hyperparameter_tuning = False
        
    # If config is None, use the default config
    elif config is None:
        run = wandb.init()
        config = run.config
        train_loader = subset_loader
        is_Hyperparameter_tuning = True
        
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
    
    # Calculate total number of steps
    total_steps = (len(train_loader)+len(val_loader)) * config.num_epochs
    pbar = tqdm(total=total_steps, desc=f'Training',  dynamic_ncols=False, ncols=50)
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0  # Initialize total_train_loss
        total_train_samples = 0  # Initialize total_train_samples
        # Calculate total number of steps
        total_steps = (len(train_loader)+len(val_loader)) * config.num_epochs
        Log_gpu_memory_usage(epoch)
        for i, (images, captions, lengths) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}", dynamic_ncols=False, ncols=50)):
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
            
            # Calculate the loss for this batch
            batch_loss = loss.item()
            # Log the batch loss to wandb
            wandb.log({"Batch Training Loss": batch_loss})
            
            pbar.update(1)
            pbar.set_description(f'Epoch {epoch + 1}/{config.num_epochs}')
            
        # Calculate and log the average training loss
        avg_train_loss = total_train_loss / total_train_samples
        wandb.log({"Epoch Training Loss": avg_train_loss,}, step=epoch)
               
        
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            total_loss = 0  # Initialize total_loss
            total_samples = 0  # Initialize total_samples
            total_correct = 0  # Initialize total_correct
            for i, (images, captions, lengths) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", dynamic_ncols=False, ncols=50)):
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
                # Calculate the loss for this batch
                batch_loss = loss.item()
                # Log the batch loss to wandb
                wandb.log({"Batch Validation Loss": batch_loss})
                pbar.update(1)
                pbar.set_description(f'Epoch {epoch + 1}/{config.num_epochs}')
                
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            print(f"Validation Loss: {avg_loss}, Accuracy: {accuracy}")

            # Log the validation loss, accuracy, and hyperparameters to wandb
            wandb.log({"Epoch Validation Loss": avg_loss,"Accuracy": accuracy,}, step=epoch)
            
            if(is_Hyperparameter_tuning == False):
                # Log a batch of images and their corresponding captions to wandb
                if epoch % 10 == 0:  # Change this to control how often you log images
                    # Get a batch of images and captions
                    images, captions, lengths = next(iter(val_loader))
                    images = images.to(device)
                    captions = captions.to(device)
                    
                    # Generate captions with the model
                    features = model.encoder(images)
                    sampled_ids = model.decoder.sample(features)
                    sampled_ids = sampled_ids.cpu().numpy()
                
                    # Convert the generated captions to strings
                    sampled_captions = []
                    for id in sampled_ids:
                        caption = ''
                        for word_id in id:
                            word = vocab.idx2word[word_id]
                            caption += word + ' '
                            if word == '<end>':
                                break
                        sampled_captions.append(caption)
                    
                    # Log the images and captions to wandb
                    wandb.log({"images": [wandb.Image(image, caption=caption) for image, caption in zip(images, sampled_captions)]})
            
            if(is_Hyperparameter_tuning == False):
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
                
            pbar.close()