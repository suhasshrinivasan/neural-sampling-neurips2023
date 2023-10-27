import os

import datajoint as dj
import torch
from gensn_experiments.monkey.likelihood.likelihood_experiment import (
    likelihood_experiment,
)

dj.config["enable_python_native_blobs"] = True

dj.config["stores"] = {
    "external": {
        "protocol": "s3",
        "endpoint": os.environ["MINIO_ENDPOINT"],
        "access_key": os.environ["MINIO_ACCESS_KEY"],
        "secret_key": os.environ["MINIO_SECRET_KEY"],
        "bucket": "neural-sampling-code",
        "location": "dj-store",
        "secure": True,
    }
}

schema = dj.schema("sshrinivasan_neurips2023")


@schema
class LikelihoodExperimentResultLog(dj.Manual):
    definition = """
    experiment_id: char(32)
    ---
    seed: int
    session_id: varchar(20)
    image_crop: int
    subsample: int
    scale: decimal(3, 2)
    batch_size: int
    n_layers: int
    nonlinearity: varchar(20)
    dropout_rate: decimal(4, 3)
    nonneg_std_transform: varchar(20)
    init_std: decimal(7, 6)
    l2_weight_decay: decimal(7, 6)
    lr: decimal(6, 5)
    n_epochs: int
    early_stopping_threshold: int
    early_stopping_patience: int
    gradient_clipping_threshold: double
    logging_type: varchar(20)
    saving_type: varchar(20)
    start_time: datetime
    model: attach@external
    train_ll: float
    val_ll: float
"""

    @staticmethod
    def get_model(key):
        return torch.load(key["model"])


@schema
class LikelihoodExperimentConfig(dj.Manual):
    definition = """
    config_id: char(32)
    ---
    seed: int
    session_id: varchar(20)
    image_crop: int
    subsample: int
    scale: float
    batch_size: int
    n_layers: int
    nonlinearity: varchar(20)
    dropout_rate: float
    nonneg_std_transform: varchar(20)
    init_std: float
    l2_weight_decay: float
    lr: float
    n_epochs: int
    early_stopping_threshold: int
    early_stopping_patience: int
    gradient_clipping_threshold: double
"""


@schema
class LikelihoodExperimentResult(dj.Computed):
    definition = """
    -> LikelihoodExperimentConfig
    ---
    model: attach@external
    train_ll: float
    val_ll: float
    """

    def make(self, key):
        args = (LikelihoodExperimentConfig & key).fetch1()
        # remove config_id from args because likelihood_experiment doesn't take it
        args.pop("config_id")
        # get device
        args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        # set logging_type
        args["logging_type"] = "stdout"
        # call likelihood_experiment with args
        model, metrics = likelihood_experiment(**args)
        # save the model
        filepath = f"/tmp/{key}.pt"
        torch.save(model, filepath)
        # insert the results
        key["model"] = filepath
        key["train_ll"] = metrics["train_ll"]
        key["val_ll"] = metrics["val_ll"]
        # insert the results into the table
        self.insert1(key)
        # remove the model from the filesystem
        os.remove(filepath)
