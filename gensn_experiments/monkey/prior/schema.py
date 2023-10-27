import os

import datajoint as dj
import torch

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
class DequantPriorExperimentResultLog(dj.Manual):
    definition = """
    experiment_id: char(32)
    ---
    session_id: varchar(20)
    image_crop: int
    subsample: int
    scale: decimal(3, 2)
    batch_size: int
    lr: decimal(6, 5)
    n_epochs: int
    prior_type: varchar(20) # prior of dequantization
    seed: int
    logging_type: varchar(20)
    saving_type: varchar(20)
    start_time: datetime
    model: attach@external
    train_ll: float
    val_ll: float
    mse_response_mean: float
    mse_response_var: float
    mse_response_kurtosis: float
    mse_response_skew: float
"""

    @staticmethod
    def get_model(key):
        return torch.load(key["model"])


@schema
class DequantPriorNeuripsResults(dj.Manual):
    definition = """
    experiment_id: char(32)
    ---
    test_ll: float  # iw bound mean per neuron, mean per trial in bits for test set
    """
