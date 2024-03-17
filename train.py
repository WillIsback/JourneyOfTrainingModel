# train.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import CNN_to_RNN
from data_preprocessing import train_loader, val_loader, vocab, subset_loader, Log_gpu_memory_usage, log_cpu_usage
import wandb
import json
from tqdm import tqdm
import Utils.Utils as Utils
from torch.cuda.amp import autocast, GradScaler 

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
# In your model or training script
vocab_size = len(vocab)

def train(config=None):
    scaler = GradScaler()
    # If config is an instance of Config, print the config and initialize wandb
    if (config):
        project="MYFIRSTAI"
        entity="william-derue"
        job_type="training"
        run = wandb.init(entity=entity, project=project, job_type=job_type)
        wandb.config.update(config)
        if run is not None:
            config = run.config
        else:
            print("wandb.init() returned None")
        is_Hyperparameter_tuning = False
        train_loader_dataset = train_loader
        
    # If config is None, use the default config
    elif config is None:
        run = wandb.init()
        if run is not None:
            config = run.config
        else:
            print("wandb.init() returned None")
        train_loader_dataset = subset_loader
        is_Hyperparameter_tuning = True
        
    # Initialize the model, loss function, and optimizer
    if config is not None:
        learning_rate = float(config.learning_rate)
        model = CNN_to_RNN(config.embed_size, config.hidden_size, vocab_size, config.num_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        print("config is None")
    criterion = nn.CrossEntropyLoss()
    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Initialize the minimum validation loss with infinity
    min_val_loss = float('inf')

    # Initialize the patience counter
    patience_counter = 0
    if config is not None:
        
        # Calculate total number of steps
        total_steps = float((len(train_loader_dataset) + len(val_loader)) * config.num_epochs)
        pbar = tqdm(total=total_steps, desc=f'Training',  dynamic_ncols=False, ncols=100)
    
        # Training loop
        for epoch in range(int(config.num_epochs)):
            model.train()
            total_train_loss = 0  # Initialize total_train_loss
            total_train_samples = 0  # Initialize total_train_samples
            # Calculate total number of steps
            total_steps = (len(train_loader_dataset)+len(val_loader)) * config.num_epochs   
            Log_gpu_memory_usage(epoch)
            log_cpu_usage(epoch)
            
            for i, (images, captions, lengths) in enumerate(tqdm(train_loader_dataset, desc=f"Training Epoch {epoch+1}", dynamic_ncols=False, ncols=100)):
                # Move data to the correct device
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                # Forward pass
                with autocast():
                    outputs = model(images, captions)
                    outputs = outputs.view(-1, vocab_size)  # Reshape outputs to [batch_size * seq_len, vocab_size]
                    outputs = outputs[:targets.size(0)]  # Remove the extra outputs due to padding
                    loss = criterion(outputs, targets)

                # Backward pass and optimization
                model.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Update total_train_loss and total_train_samples
                total_train_loss += loss.item() * images.size(0)
                total_train_samples += targets.size(0)
                
                # Calculate the loss for this batch
                batch_loss = loss.item()
                # Log the step to wandb
                wandb.log({"Training Batch Loss": batch_loss})
                
                pbar.update(1)
                pbar.set_description(f'Epoch {epoch + 1}/{config.num_epochs}')
                
            # Calculate and log the average training loss
            avg_train_loss = total_train_loss / total_train_samples
            wandb.log({"Training Loss": avg_train_loss,})
            pbar.update(1)
            
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                total_loss = 0  # Initialize total_loss
                total_samples = 0  # Initialize total_samples
                total_correct = 0  # Initialize total_correct
                for i, (images, captions, lengths) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", dynamic_ncols=False, ncols=100)):
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
                    # Log the step to wandb
                    wandb.log({"Validation batch Loss": batch_loss})
                    
                    pbar.update(1)              
                avg_loss = total_loss / total_samples
                accuracy = total_correct / total_samples
                print(f"Validation Loss: {avg_loss}, Accuracy: {accuracy}")
                pbar.update(1)
                # Log the validation loss, accuracy, and hyperparameters to wandb
                wandb.log({"Validation Loss": avg_loss,"Accuracy": accuracy,})
                
                if(is_Hyperparameter_tuning == False):
                    # Log a batch of images and their corresponding captions to wandb
                    if (epoch + 1) % 10 == 2:  # Log images on the 2nd, 12th, 22nd epoch, etc.
                        # Get a batch of images and captions
                        images, captions, lengths = next(iter(val_loader))
                        images = images[:3].to(device)  # Select only the first 3 images
                        captions = captions[:3].to(device)  # Select only the first 3 captions
                        
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
    else:
        print("config is None")