import argparse
import os
from datetime import datetime
from time import time

import torch
from likelihood_experiment import likelihood_experiment

import wandb

# TODO: write dj schema for this
# from gensn_experiments.monkey.likelihood.schema import LikelihoodExperimentResultLog
from neural_sampling_code.utilities.utilities import make_hash

parser = argparse.ArgumentParser()
# parse data params
parser.add_argument("--session_id", type=str, default="3631807112901")
parser.add_argument("--image_crop", default=96, type=int)
parser.add_argument("--subsample", default=1, type=int)
parser.add_argument("--scale", default=1.0, type=float)
parser.add_argument("--batch_size", default=128, type=int)
# parse model params
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--nonlinearity", type=str, default="relu")
parser.add_argument("--dropout_rate", type=float, default=0.0)
parser.add_argument("--nonneg_std_transform", type=str, default="exp")
parser.add_argument("--init_std", type=float, default=1e-3)
# parse training params
parser.add_argument("--l2_weight_decay", type=float, default=0)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--n_epochs", default=100, type=int)
parser.add_argument("--early_stopping_threshold", default=3, type=int)
parser.add_argument("--early_stopping_patience", default=3, type=int)
parser.add_argument("--gradient_clipping_threshold", default=1e6, type=float)
# parse misc params
parser.add_argument("--saving_type", default="dj", type=str)
parser.add_argument("--logging_type", default="wandb", type=str)
parser.add_argument("--seed", default=42, type=int)


def save_experiment(model, metrics, config, unique_experiment_id):
    # Put the model on the cpu before saving
    model.to("cpu")
    # Merge metrics and config to create a single dict to save
    results = {**config, **metrics}

    # Add the unique experiment id to the results dict
    results["experiment_id"] = unique_experiment_id

    # if saving locally, add the model to the results dict
    # and save the results dict to a file
    if config["saving_type"] == "file":
        filepath = f"/src/project/computed/{unique_experiment_id}.pt"
        results["model"] = model
        torch.save(results, filepath)

    elif config["saving_type"] == "dj":

        # dynamic import to avoid running dj code if not saving to dj
        from gensn_experiments.monkey.likelihood.schema import (
            LikelihoodExperimentResultLog,
        )

        # Save the model to tmp and push filename to results
        # because the LikelihoodExperimentResultLog
        # table will automatically fetch the model from tmp and save it
        # on s3
        filepath = f"/tmp/{unique_experiment_id}.pt"
        torch.save(model, filepath)

        # Add the filepath of the save model to results
        results["model"] = filepath

        # Insert tuple and remove the model in tmp since it is uploaded to S3
        LikelihoodExperimentResultLog.insert1(results, ignore_extra_fields=True)
        os.remove(filepath)


def main():
    # Parse args
    config = vars(parser.parse_args())

    # Set datapath
    data_basepath = "/src/project/computed/compiled_monkey_datasets/"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Update config with device and data_basepath
    config.update({"device": device, "data_basepath": data_basepath})

    # Add start time to config
    config.update({"start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    # Create unique experiment id
    unique_experiment_id = make_hash(config)  # 32 char hash

    # if logging to wandb, initialize wandb
    # log the config and save the unique experiment id as a tag
    if config["logging_type"] == "wandb":
        wandb.init(
            project="monkey_decoding_with_gensn",
            entity="walkerlab",
            config=config,
            tags=[unique_experiment_id],
        )

    # Run experiment
    # first remove dict elements that are not used by the likelihood_experiment function
    experiment_config = config.copy()
    experiment_config.pop("saving_type")
    experiment_config.pop("start_time")

    # Call experiment
    model, metrics = likelihood_experiment(**experiment_config)

    # Save experiment results
    save_experiment(model, metrics, config, unique_experiment_id)


if __name__ == "__main__":
    main()
