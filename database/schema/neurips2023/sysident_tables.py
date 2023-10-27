import os

import datajoint as dj
import torch
from system_identification.monkey_sysident_experiment import monkey_sysident_experiment

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
class MonkeySysidentConfig2(dj.Manual):
    definition = """
    # Config table to train monkey system identification models
    # for the neural sampling project, NeurIPS 2023
    config_id: char(32)
    ---
    seed: int   # random seed
    session_id: varchar(20)
    image_crop: int
    subsample: int
    scale: float
    batch_size: int
    time_sum_bins: int
    gamma_eps: float    # gamma distribution epsilon
    gamma_min_rate: float   # gamma distribution min rate
    gamma_max_concentration: float  # gamma distribution max concentration
    model_config: longblob  
    trainer_config: longblob
    """


@schema
class MonkeySysidentConfig3(dj.Manual):
    definition = """
    # Config table to train monkey system identification models
    # for the neural sampling project, NeurIPS 2023
    config_id: char(32)
    ---
    seed: int   # random seed
    session_id: varchar(20)
    image_crop: int
    subsample: int
    scale: float
    batch_size: int
    time_sum_bins: int
    gamma_eps: float    # gamma distribution epsilon
    gamma_min_rate: float   # gamma distribution min rate
    gamma_max_concentration: float  # gamma distribution max concentration
    model_config: longblob  
    trainer_config: longblob
    """


@schema
class MonkeySysidentResults2(dj.Computed):
    definition = """
    # Results table to train and save monkey system identification models
    # and their performance for the neural sampling project, NeurIPS 2023
    -> MonkeySysidentConfig2
    ---
    model_state_dict: attach@external
    train_pmf: float
    val_pmf: float
    test_pmf: float
    train_ll: float
    val_ll: float
    test_ll: float
    train_correlation: float
    validation_correlation: float
    test_correlation: float
    trainer_output: longblob
    """

    def make(self, key):
        args = (MonkeySysidentConfig2 & key).fetch1()
        args.pop("config_id")
        model_state_dict, performance, trainer_output = monkey_sysident_experiment(
            **args
        )

        filepath = f"/tmp/{key}.pt"
        torch.save(model_state_dict, filepath)

        key["model_state_dict"] = filepath
        key["trainer_output"] = trainer_output
        key.update(performance)

        self.insert1(key, skip_duplicates=True)

        os.remove(filepath)


@schema
class MonkeySysidentResults3(dj.Computed):
    definition = """
    # Results table to train and save monkey system identification models
    # and their performance for the neural sampling project, NeurIPS 2023
    -> MonkeySysidentConfig3
    ---
    model_state_dict: attach@external
    train_pmf: float
    val_pmf: float
    test_pmf: float
    train_ll: float
    val_ll: float
    test_ll: float
    train_correlation: float
    validation_correlation: float
    test_correlation: float
    trainer_output: longblob
    """

    def make(self, key):
        args = (MonkeySysidentConfig3 & key).fetch1()
        args.pop("config_id")
        model_state_dict, performance, trainer_output = monkey_sysident_experiment(
            **args
        )

        filepath = f"/tmp/{key}.pt"
        torch.save(model_state_dict, filepath)

        key["model_state_dict"] = filepath
        key["trainer_output"] = trainer_output
        key.update(performance)

        self.insert1(key, skip_duplicates=True)

        os.remove(filepath)


@schema
class MonkeySysidentGeneralPosteriorConfig(dj.Manual):
    definition = """  
    # Config table to train monkey system identification models
    # with general posterior model type
    config_id: char(32)
    ---
    seed: int   # random seed
    session_id: varchar(20)
    image_crop: int 
    subsample: int  # subsample factor of the image
    scale: float    # scale factor of the image
    batch_size: int
    time_sum_bins: int  # number of time bins to sum over spike counts
    distribution_type: varchar(20) # type of sysident distribution, i.e., "gamma", or "poisson", ...
    model_config: longblob  # config for the system identification model
    trainer_config: longblob    # config for the trainer (generally nnsysident trainer)
    """


@schema
class MonkeySysidentGeneralPosteriorResults(dj.Computed):
    definition = """
    # Results table to train and save monkey system identification models
    # and their performance for the neural sampling project
    -> MonkeySysidentGeneralPosteriorConfig
    ---
    model_state_dict: attach@external
    train_pmf_sum: float    # sum over neurons, mean per trial, in bits
    train_pmf_mean: float   # mean over neurons, mean per trial, in bits
    train_pmf_sem: float    # sde over neurons, mean per trial, in bits
    val_pmf_sum: float    # sum over neurons, mean per trial, in bits
    val_pmf_mean: float   # mean over neurons, mean per trial, in bits
    val_pmf_sem: float    # sde over neurons, mean per trial, in bits
    test_pmf_sum: float   # sum over neurons, mean per trial, in bits
    test_pmf_mean: float  # mean over neurons, mean per trial, in bits
    test_pmf_sem: float   # sde over neurons, mean per trial, in bits
    train_correlation: float    # mean per neuron, mean per trial
    train_correlation_sem: float    # sde per neuron, mean per trial
    validation_correlation: float   # mean per neuron, mean per trial
    validation_correlation_sem: float   # sde per neuron, mean per trial
    test_correlation: float   # mean per neuron, mean per trial
    test_correlation_sem: float   # sde per neuron, mean per trial
    trainer_output: longblob    # output of the trainer
    """

    def make(self, key):
        args = (MonkeySysidentGeneralPosteriorConfig & key).fetch1()
        args.pop("config_id")
        model_state_dict, performance, trainer_output = monkey_sysident_experiment(
            **args
        )

        filepath = f"/tmp/{key}.pt"
        torch.save(model_state_dict, filepath)

        key["model_state_dict"] = filepath
        key["trainer_output"] = trainer_output
        key.update(performance)

        self.insert1(key, skip_duplicates=True)

        os.remove(filepath)


@schema
class MonkeySysidentNeuripsResults(dj.Manual):
    """
    This table consists of the results from the best models from the sysident experiments on monkey
    """

    definition = """
    config_id: char(32)
    ---
    test_post_ll: float    # mean per neuron, mean per trial, in bits
    test_post_ll_sem: float      # sde per neuron

    test_correlation: float # mean per neuron, mean per trial
    test_correlation_sem: float # sde per neuron
    """


@schema
class MonkeySysidentNeuripsResultsRiemann(dj.Manual):
    """
    This table consists of the results from the best models from MonkeySysidentConfig3
    specifically, it stores log pmf in addition to log likelihood, computed using riemann sum
    """

    definition = """
    config_id: char(32)
    ---
    test_post_ll: float    # mean per neuron, mean per trial, in bits
    test_post_ll_sem: float      # sde per neuron

    test_post_pmf: float    # mean per neuron, mean per trial, in bits, riemann sum
    test_post_pmf_sem: float      # sde per neuron

    test_correlation: float # mean per neuron, mean per trial
    test_correlation_sem: float # sde per neuron
    """
