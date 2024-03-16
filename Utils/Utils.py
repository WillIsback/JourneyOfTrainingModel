import json

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
                'values': [3, 5, 7] # [3, 5, 7]
            },
            'num_epochs': {
                'values': [5, 10, 15] # [5, 10, 15]
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'max_iter': 15,
            'min_iter': 5,
            'eta': 3,
        }
    }
    # Write the sweep_config dictionary to a JSON file
    with open('configs/sweep_config.json', 'w') as f:
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