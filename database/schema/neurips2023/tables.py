import os

import datajoint as dj

dj.config["enable_python_native_blobs"] = True

schema = dj.schema("sshrinivasan_neurips2023")

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


@schema
class DequantPriorExperiment(dj.Manual):
    definition = """
    session_id: varchar(20)
    image_crop: int
    subsample: int
    scale: decimal(2, 2)
    batch_size: int
    lr: decimal(6, 5)
    n_epochs: int
    prior_type: varchar(20)
    seed: int
    logging_type: varchar(20)
    saving_type: varchar(20)
    time: datetime
    ---
    experiment_id: varchar(50)
    model: attach@external
    train_ll: float
    val_ll: float
    test_ll: float
    mse_response_mean: float
    mse_response_var: float
    mse_response_kurtosis: float
    mse_response_skewness: float
    """
