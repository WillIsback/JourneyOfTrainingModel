# main.py
from train import train
from evaluate import evaluate
import torch
import torch.nn as nn
from model import CNN_to_RNN
from data_preprocessing import load_and_preprocess_data
import json
from Utils.MultiProcessingSweep import RunMultiProcessingSweep
from Utils.Utils import sweep_config_maker, get_absolute_path
import Utils.Utils as Utils

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(device.type == 'cuda'):
    print(f"Using the {device} for training")
else:
    print(f"Using the {device} for training")
    
# Ask the user if they want to fine-tune the hyperparameters
fine_tune = input("Do you want to fine-tune the hyperparameters? (yes/no): ")

if fine_tune.lower() == 'yes':
    HowManySweepRuns = int(input("How many sweep runs do you want to run?: "))

# Load and preprocess the data, and retrieve the vocabulary
_, _, vocab = load_and_preprocess_data()

if __name__ == '__main__':
    if fine_tune.lower() == 'yes':
        
        sweep_config_maker() # Create the sweep configuration json file
        
        # Load the sweep configuration from the JSON file
        with open(get_absolute_path('configs/sweep_config.json'), 'r') as f:
            sweep_config_dict = json.load(f)
            
        # Run the sweep
        RunMultiProcessingSweep(sweep_config_dict, HowManySweepRuns)  # Call the run_sweep function with your sweep_config

        # Load the wandb sweep best hyperparameters from the JSON file
        with open(get_absolute_path('configs/best_config.json'), 'r') as f:
            config_dict = json.load(f)
            config = Utils.Config(config_dict)
    else:
        # Load the manually choosen hyperparameters from the JSON file
        with open(get_absolute_path('configs/hardconfig.json'), 'r') as f:
            config_dict = json.load(f)
            config = Utils.Config(config_dict)

    # Train the model with the chosen hyperparameters
    train(config)

    # Load the best trained model
    model = CNN_to_RNN(config['embed_size'], config['hidden_size'], len(vocab), config['num_layers']).to(device)
    model.load_state_dict(torch.load('model.pt'))

    # Evaluate the model on the test set
    evaluate(model, get_absolute_path('DataSet/test2017'), device)