# main.py
from model import CNN_Encoder, RNN_Decoder
from train import train
from evaluate import evaluate
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import CNN_to_RNN
from data_preprocessing import test_loader, vocab
import wandb
import json

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In your model or training script
vocab_size = len(vocab)

# Define your hyperparameters
sweep_config = {
    'method': 'random',  # grid, random
    'metric': {
      'name': 'Validation Loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'embed_size': {
            'values': [256, 512, 1024]
        },
        'hidden_size': {
            'values': [512, 1024, 2048]
        },
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01
        },
        'num_layers': {
            'values': [1, 2, 3]
        },
        'patience': {
            'values': [2, 4, 6] # [3, 5, 7]
        },
        'num_epochs': {
            'values': [1, 2, 3] # 5 10 15
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="MYFIRSTAI", entity="william-derue")

# Use wandb.agent() to execute the train() function for each run of the sweep
wandb.agent(sweep_id, function=train)

# Load the config object from the file
with open('config.json', 'r') as f:
    config = json.load(f)

# Convert the config values back to the correct types
config = wandb.config.update(config, allow_val_change=True)

# Load the trained model
model = CNN_to_RNN(config.embed_size, config.hidden_size, vocab_size, config.num_layers).to(device)
model.load_state_dict(torch.load('model.pt'))

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Evaluate the model on the test set
test_loss, test_accuracy = evaluate(model, test_loader, criterion, config)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")