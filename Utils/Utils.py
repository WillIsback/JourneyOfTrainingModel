import json
from pathlib import Path
import torch
import psutil   
import logging
import subprocess

class Config: # This class is used to store the hyperparameters and convert them from a dictionary to an object
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
def get_serializable_config(config):
    serializable_config = {}
    for key, value in vars(config).items():
        try:
            json.dumps(value)  # Try to serialize the value to JSON
            serializable_config[key] = value  # If no error is raised, add the item to the dictionary
        except TypeError:
            pass  # If a TypeError is raised, ignore the item
    return serializable_config


def sweep_config_maker():
    sweep_config = {
        'method': 'random',  # grid, random
        'metric': {
        'name': 'Validation Loss',
        'goal': 'minimize'   
        },
        'parameters': {
            'embed_size': {
                'values': [512, 1024, 2048]
            },
            'hidden_size': {
                'values': [1024, 2048, 4096]
            },
            'learning_rate': {
                'min': 0.00001,
                'max': 0.001 # [0.0001, 0.001, 0.01]
            },
            'num_layers': {
                'values': [2, 3, 4]
            },
            'patience': {
                'values': [1, 3, 5] # [3, 5, 7]
            },
            'num_epochs': {
                'values': [1, 3, 5] # [5, 10, 15]
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'max_iter': 5,
            'min_iter': 1,
            'eta': 3,
        }
    }
    # Write the sweep_config dictionary to a JSON file
    with open(get_absolute_path('configs/sweep_config.json'), 'w') as f:
        json.dump(sweep_config, f)
        
def PrintNumberOfSweep(sweep_config):
    # Calculate the total number of combinations
    total_combinations = len(sweep_config['parameters']['embed_size']['values']) * \
                        len(sweep_config['parameters']['hidden_size']['values']) * \
                        sweep_config['early_terminate']['max_iter'] * \
                        len(sweep_config['parameters']['num_layers']['values']) * \
                        len(sweep_config['parameters']['patience']['values']) * \
                        len(sweep_config['parameters']['num_epochs']['values'])
                        
    print(f"Total number of combinations: {total_combinations}")
    
    
def get_absolute_path(relative_path):
    root_dir = Path(__file__).resolve().parent
    while not (root_dir / '.gitignore').exists():  # replace '.git' with your marker file or directory
        root_dir = root_dir.parent
    return root_dir / relative_path


def CheckingDataSetSetup(train_loader_dataset, val_loader_dataset):
    # Check if the data loaders are correctly set up
    try:
        train_images, train_captions, train_lengths = next(iter(train_loader_dataset))
        val_images, val_captions, val_lengths = next(iter(val_loader_dataset))
    except Exception as e:
        print(f"Error while loading data: {e}")
        exit(1)

    # Check if the batches have the expected structure
    if not (isinstance(train_images, torch.Tensor) and isinstance(train_captions, torch.Tensor) and isinstance(train_lengths, torch.Tensor)):
        print("Error: Unexpected structure of training batch")
        exit(1)

    if not (isinstance(val_images, torch.Tensor) and isinstance(val_captions, torch.Tensor) and isinstance(val_lengths, torch.Tensor)):
        print("Error: Unexpected structure of validation batch")
        exit(1)
       

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(f"Free memory: {memory_free_values}")
    return memory_free_values
 
def Log_gpu_memory_usage(epoch):
    memory_used = get_gpu_memory()
    logging.info(f'Epoch {epoch}, Memory used: {memory_used} MB')
    
def log_cpu_usage(epoch):
    cpu_usage = psutil.cpu_percent()
    logging.info(f'Epoch {epoch}, CPU usage: {cpu_usage}%')