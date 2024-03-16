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

# Ask the user if they want to fine-tune the hyperparameters
fine_tune = input("Do you want to fine-tune the hyperparameters? (yes/no): ")

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
            'min': 0.00001,
            'max': 0.0001 # [0.0001, 0.001, 0.01]
        },
        'num_layers': {
            'values': [1, 2, 3]
        },
        'patience': {
            'values': [2, 3, 4] # [3, 5, 7]
        },
        'num_epochs': {
            'values': [4, 6, 8] # [5, 10, 15]
        }
    },
    
    'early_terminate': {
        'type': 'hyperband',
        's': 2,
        'max_iter': 10,
        'eta': 3,
    }
}
# Calculate the total number of combinations
total_combinations = len(sweep_config['parameters']['embed_size']['values']) * \
                     len(sweep_config['parameters']['hidden_size']['values']) * \
                     10 * \
                     len(sweep_config['parameters']['num_layers']['values']) * \
                     len(sweep_config['parameters']['patience']['values']) * \
                     len(sweep_config['parameters']['num_epochs']['values'])
                     
if fine_tune.lower() == 'yes':
    # Run the sweep
    sweep_id = wandb.sweep(sweep_config, project="MYFIRSTAI", entity="william-derue")
    wandb.agent(sweep_id, function=train)

    # Load the hyperparameters from the JSON file
    with open('config.json', 'r') as f:
        config = json.load(f)
else:
    # Use the hard-coded best hyperparameters
    config = {
        'embed_size': 512,
        'hidden_size': 1024,
        'learning_rate': 0.0001073,
        'num_layers': 2,
        'patience': 3,
        'num_epochs': 4
    }

# Train the model with the chosen hyperparameters
train(config)

# Load the best trained model
model = CNN_to_RNN(config['embed_size'], config['hidden_size'], vocab_size, config['num_layers']).to(device)
model.load_state_dict(torch.load('model.pt'))

# Define the loss function
criterion = nn.CrossEntropyLoss()


# Evaluate the model on the test set
test_loss, test_accuracy = evaluate(model, test_loader, criterion, config)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")