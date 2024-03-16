from train import train # Import the train function from train.py
from wandb.sdk import wandb_config
import wandb
import json
import Utils.Utils as Utils
from tqdm import tqdm


def run_sweep(sweep_id, count):
    # Initialize the progress bar
    pbar = tqdm(total=count, desc='Overall Progress', dynamic_ncols=False, ncols=100)
    def train_with_progress(*args, **kwargs):
        # Run the training function and measure the time
        train(*args, **kwargs)
        # Update the progress bar
        pbar.update(1)

    wandb.agent(sweep_id, function=train_with_progress, count=count)
    GetBestSweepRun(sweep_id)

    # Close the progress bar
    pbar.close()
    
def RunMultiProcessingSweep(sweep_config, count=1):
    sweep_id = wandb.sweep(sweep_config, project="MYFIRSTAI", entity="william-derue")
    Utils.PrintNumberOfSweep(sweep_config)
    run_sweep(sweep_id, count)
        
def GetBestSweepRun(sweep_id):
    # Initialize the wandb API
    api = wandb.Api()
    # sweep id
    sweep = api.sweep(sweep_id)
    # Get the best run of the sweep
    best_run = sorted(sweep.runs, key=lambda run: run.summary.get('Average Validation Loss', 0))[0]
    # Write the best run's hyperparameters to a JSON file
    with open('configs/best_config.json', 'w') as f:
        json.dump(best_run.config, f)
        
