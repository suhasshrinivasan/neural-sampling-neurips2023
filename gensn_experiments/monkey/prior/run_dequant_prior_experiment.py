import argparse
import os
from datetime import datetime
from time import time

import torch
from dequant_prior_experiment import dequant_prior_comparison_experiment
from neural_sampling_code.utilities.utilities import make_hash

import wandb

parser = argparse.ArgumentParser()

parser.add_argument("--session_id", type=str, default="3631807112901")
parser.add_argument("--image_crop", default=96, type=int)
parser.add_argument("--subsample", default=1, type=int)
parser.add_argument("--scale", default=1.0, type=float)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--n_epochs", default=5, type=int)
parser.add_argument("--prior_type", default="exp", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--logging_type", default="wandb", type=str)
parser.add_argument("--saving_type", default="dj", type=str)


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
        from gensn_experiments.monkey.prior.schema import (
            DequantPriorExperimentResultLog,
        )

        # Save the model to tmp and push filename to results
        # because the DequantPriorExperimentResultLog
        # table will automatically fetch the model from tmp and save it
        # on s3
        filepath = f"/tmp/{unique_experiment_id}.pt"
        torch.save(model, filepath)

        # Add the filepath of the save model to results
        results["model"] = filepath

        # Insert tuple and remove the model in tmp since it is uploaded to S3
        DequantPriorExperimentResultLog.insert1(results, ignore_extra_fields=True)
        os.remove(filepath)


def main():
    print("Parsing args and preparing config")
    # create a config dict from the parsed args
    config = vars(parser.parse_args())
    data_basepath = "/data/monkey/toliaslab/compiled_monkey_datasets/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # update config with additional information
    config.update(
        dict(
            data_basepath=data_basepath,
            device=device,
        )
    )
    # use config hash and time to make unique id for the experiment
    config["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    unique_experiment_id = make_hash(config)  # 32 char hash
    # if logging to wandb, initialize wandb
    # and add the unique id as a tag
    if config["logging_type"] == "wandb":
        wandb.init(
            project="spike_count_dequant",
            entity="walkerlab",
            config=config,
            tags=[unique_experiment_id],
        )
    print("Calling dequant_prior_comparison_experiment with config")
    training_config = config.copy()
    training_config.pop("start_time")
    training_config.pop("saving_type")
    model, metrics = dequant_prior_comparison_experiment(**training_config)
    print("Saving model and metrics")
    save_experiment(model, metrics, config, unique_experiment_id)


if __name__ == "__main__":
    main()
