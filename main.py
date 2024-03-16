# main.py
from train import train
from evaluate import evaluate
import torch
import torch.nn as nn
from model import CNN_to_RNN
from data_preprocessing import test_loader, vocab
import json
from Utils.MultiProcessingSweep import RunMultiProcessingSweep
from Utils.Utils import sweep_config_maker
import Utils.Utils as Utils
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ask the user if they want to fine-tune the hyperparameters
fine_tune = input("Do you want to fine-tune the hyperparameters? (yes/no): ")

if fine_tune.lower() == 'yes':
    HowManySweepRuns = int(input("How many sweep runs do you want to run?: "))


# In your model or training script
vocab_size = len(vocab)



if __name__ == '__main__':
    if fine_tune.lower() == 'yes':
        
        sweep_config_maker() # Create the sweep configuration json file
        
        # Load the sweep configuration from the JSON file
        with open('configs/sweep_config.json', 'r') as f:
            sweep_config_dict = json.load(f)
            
        # Run the sweep
        RunMultiProcessingSweep(sweep_config_dict, HowManySweepRuns)  # Call the run_sweep function with your sweep_config

        # Load the wandb sweep best hyperparameters from the JSON file
        with open('configs/best_config.json', 'r') as f:
            config_dict = json.load(f)
            config = Utils.Config(config_dict)
    else:
        # Load the manually choosen hyperparameters from the JSON file
        with open('configs/hardconfig.json', 'r') as f:
            config_dict = json.load(f)
            config = Utils.Config(config_dict)

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