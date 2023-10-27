import os
import tempfile

import datajoint as dj
import numpy as np
import torch
from database.schema.neurips2023.load_from_dj import load_best_sampling_row
from database.schema.neurips2023.posterior_schema import (
    SampleBasedNNSysIdentResults2,
    SampleBasedSysIdentConfig2,
)
from neural_sampling_code.elements.evaluation import (
    marginal_likelihood,
    sysident_correlation,
    sysident_log_pmf_riemann,
    sysident_log_pmf_uniform,
)
from neural_sampling_code.utilities.utilities import load_compiled_monkey_dataset

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


config_table_class_map = {
    "SampleBasedSysIdentConfig2": SampleBasedSysIdentConfig2,
}
result_table_class_map = {
    "SampleBasedNNSysIdentResults2": SampleBasedNNSysIdentResults2,
}


def fill_dict_with_sysident_metrics(
    dict, config, model, train_loader, validation_loader, test_loader, device
):
    """
    This function fills the passed in dictionary with the following metrics:
    - train_iwu_pmf_sum: float    # log pmf(r|x) using iwu, sum over neurons, mean over trial, in bits
    - train_iwu_pmf_mean: float   # log pmf(r|x) using iwu, mean over neurons, mean over trial, in bits
    - train_iwu_pmf_sem: float    # log pmf(r|x) using iwu, sde over neurons, mean over trial, in bits
    - validation_iwu_pmf_sum: float    # log pmf(r|x) using iwu, sum over neurons, mean over trial, in bits
    - validation_iwu_pmf_mean: float   # log pmf(r|x) using iwu, mean over neurons, mean over trial, in bits
    - validation_iwu_pmf_sem: float    # log pmf(r|x) using iwu, sde over neurons, mean over trial, in bits
    - test_iwu_pmf_sum: float   # log pmf(r|x) using iwu, sum over neurons, mean over trial, in bits
    - test_iwu_pmf_mean: float  # log pmf(r|x) using iwu, mean over neurons, mean over trial, in bits
    - test_iwu_pmf_sem: float   # log pmf(r|x) using iwu, sde over neurons, mean over trial, in bits

    - train_rs_pmf_sum: float    # log pmf(r|x) using rs, sum over neurons, mean over trial, in bits
    - train_rs_pmf_mean: float   # log pmf(r|x) using rs, mean over neurons, mean over trial, in bits
    - train_rs_pmf_sem: float    # log pmf(r|x) using rs, sde over neurons, mean over trial, in bits
    - validation_rs_pmf_sum: float    # log pmf(r|x) using rs, sum over neurons, mean over trial, in bits
    - validation_rs_pmf_mean: float   # log pmf(r|x) using rs, mean over neurons, mean over trial, in bits
    - validation_rs_pmf_sem: float    # log pmf(r|x) using rs, sde over neurons, mean over trial, in bits
    - test_rs_pmf_sum: float   # log pmf(r|x) using rs, sum over neurons, mean over trial, in bits
    - test_rs_pmf_mean: float  # log pmf(r|x) using rs, mean over neurons, mean over trial, in bits
    - test_rs_pmf_sem: float   # log pmf(r|x) using rs, sde over neurons, mean over trial, in bits

    - train_correlation: float  # correlation between true and estimated posterior mean, mean over neurons, mean over trials
    - train_correlation_sem: float  # correlation between true and estimated posterior mean, sde over neurons, mean over trials
    - validation_correlation: float  # correlation between true and estimated posterior mean, mean over neurons, mean over trials
    - validation_correlation_sem: float  # correlation between true and estimated posterior mean, sde over neurons, mean over trials
    - test_correlation: float  # correlation between true and estimated posterior mean, mean over neurons, mean over trials
    - test_correlation_sem: float  # correlation between true and estimated posterior mean, sde over neurons, mean over trials

    Args:
        dict: dictionary to fill
        config: config to use for computing metrics such as n_samples to compute pmf
        model: model to use for computing metrics
        train_loader: train data loader
        validation_loader: validation data loader
        test_loader: test data loader
        device: device to use for computing metrics

    Returns:
        None

    Remarks:
        This function was written to to be used to fill metrics for the following dj.Computed tables:
        - PosteriorPMFApproxResult
        - SysidentPMFApproxResult
    """
    tiers = ["train", "validation", "test"]

    # compute pmf using importance weighted sampling
    print("Computing pmf using importance weighted sampling")
    iwu_pmfs = [
        sysident_log_pmf_uniform(
            model=model,
            dataloader=dataloader,
            n_samples=config["n_samples"],
            device=device,
        )
        for dataloader in [train_loader, validation_loader, test_loader]
    ]
    for tier, iwu_pmf in zip(tiers, iwu_pmfs):
        dict[f"{tier}_iwu_pmf_sum"] = iwu_pmf[0]
        dict[f"{tier}_iwu_pmf_mean"] = iwu_pmf[1]
        dict[f"{tier}_iwu_pmf_sem"] = iwu_pmf[2]

    # compute pmf using riemann sum
    print("Computing pmf using riemann sum")
    rs_pmfs = [
        sysident_log_pmf_riemann(
            model=model,
            dataloader=dataloader,
            n_samples=config["n_samples"],
            device=device,
        )
        for dataloader in [train_loader, validation_loader, test_loader]
    ]
    for tier, rs_pmf in zip(tiers, rs_pmfs):
        dict[f"{tier}_rs_pmf_sum"] = rs_pmf[0]
        dict[f"{tier}_rs_pmf_mean"] = rs_pmf[1]
        dict[f"{tier}_rs_pmf_sem"] = rs_pmf[2]

    # compute correlation between true and estimated posterior mean
    print("Computing correlation between true and estimated posterior mean")
    correlations = [
        sysident_correlation(
            model=model,
            dataloader=dataloader,
            device=device,
            distribution=config["posterior_distribution"],
        )
        for dataloader in [train_loader, validation_loader, test_loader]
    ]
    for tier, correlation in zip(tiers, correlations):
        dict[f"{tier}_correlation"] = correlation[0]
        dict[f"{tier}_correlation_sem"] = correlation[1]


@schema
class PosteriorPMFApproxConfig(dj.Manual):
    """
    Table that forms the config for computing pmf of discrete spike counts under a trained continuous distribution model.
    The PMF is computed using two methods:
    1. Importance weighted sampling using uniform proposal distribution (iwu)
    2. Riemann sum (rs)"""

    definition = """
    config_id: varchar(32) # config id
    ---
    seed: int # seed
    session_id: varchar(32) # session id
    image_crop: int # image crop
    subsample: int # subsample
    scale: float # scale
    batch_size: int # batch size
    n_samples: int # number of samples used to compute the pmf
    prior_model_type: varchar(32) # prior model type
    likelihood_model_type: varchar(32) # likelihood model type
    posterior_distribution: varchar(32) # posterior distribution
    config_table_name: varchar(100) # name of the config table that stores the config of the trained continuous model.
    result_table_name: varchar(100) # name of the result table that stores the trained continuous model.
    """


@schema
class PosteriorPMFApproxResult(dj.Computed):
    """
    Table that stores the pmf of discrete spike counts under a continuous distribution.
    The PMF is computed using two methods:
    1. Importance weighted sampling using uniform proposal distribution (iwu)
    2. Riemann sum (rs)
    Since along with pmf, correlation is also a measure of interest, it is also computed and stored.
    """

    data_basepath = "/data/monkey/toliaslab/compiled_monkey_datasets/"
    device = "cpu"

    definition = """
    -> PosteriorPMFApproxConfig
    ---
    train_iwu_pmf_sum: float    # log pmf(r|x) using iwu, sum over neurons, mean over trial, in bits
    train_iwu_pmf_mean: float   # log pmf(r|x) using iwu, mean over neurons, mean over trial, in bits
    train_iwu_pmf_sem: float    # log pmf(r|x) using iwu, sde over neurons, mean over trial, in bits
    validation_iwu_pmf_sum: float    # log pmf(r|x) using iwu, sum over neurons, mean over trial, in bits
    validation_iwu_pmf_mean: float   # log pmf(r|x) using iwu, mean over neurons, mean over trial, in bits
    validation_iwu_pmf_sem: float    # log pmf(r|x) using iwu, sde over neurons, mean over trial, in bits
    test_iwu_pmf_sum: float   # log pmf(r|x) using iwu, sum over neurons, mean over trial, in bits
    test_iwu_pmf_mean: float  # log pmf(r|x) using iwu, mean over neurons, mean over trial, in bits
    test_iwu_pmf_sem: float   # log pmf(r|x) using iwu, sde over neurons, mean over trial, in bits

    train_rs_pmf_sum: float    # log pmf(r|x) using rs, sum over neurons, mean over trial, in bits
    train_rs_pmf_mean: float   # log pmf(r|x) using rs, mean over neurons, mean over trial, in bits
    train_rs_pmf_sem: float    # log pmf(r|x) using rs, sde over neurons, mean over trial, in bits
    validation_rs_pmf_sum: float    # log pmf(r|x) using rs, sum over neurons, mean over trial, in bits
    validation_rs_pmf_mean: float   # log pmf(r|x) using rs, mean over neurons, mean over trial, in bits
    validation_rs_pmf_sem: float    # log pmf(r|x) using rs, sde over neurons, mean over trial, in bits
    test_rs_pmf_sum: float   # log pmf(r|x) using rs, sum over neurons, mean over trial, in bits
    test_rs_pmf_mean: float  # log pmf(r|x) using rs, mean over neurons, mean over trial, in bits
    test_rs_pmf_sem: float   # log pmf(r|x) using rs, sde over neurons, mean over trial, in bits

    train_correlation: float  # correlation between true and estimated posterior mean, mean over neurons, mean over trials
    train_correlation_sem: float  # correlation between true and estimated posterior mean, sde over neurons, mean over trials
    validation_correlation: float  # correlation between true and estimated posterior mean, mean over neurons, mean over trials
    validation_correlation_sem: float  # correlation between true and estimated posterior mean, sde over neurons, mean over trials
    test_correlation: float  # correlation between true and estimated posterior mean, mean over neurons, mean over trials
    test_correlation_sem: float  # correlation between true and estimated posterior mean, sde over neurons, mean over trials

    config_table_id: char(32)   # id of the config table that stores the config of the trained continuous model.
    """

    def make(self, key):
        print("Populating PosteriorPMFApproxResult")
        print("Arguments in use:")
        print(f"Device: {self.device}")
        print(f"Data basepath: {self.data_basepath}")

        config = (PosteriorPMFApproxConfig & key).fetch1()

        config_table_class = config_table_class_map[config["config_table_name"]]
        result_table_class = result_table_class_map[config["result_table_name"]]
        print(f"Config table: {config_table_class.__name__}")
        print(f"Result table: {result_table_class.__name__}")

        # fetch best model for given config
        print("Fetching best model for given config")
        row = load_best_sampling_row(
            session_id=config["session_id"],
            prior_model_type=config["prior_model_type"],
            likelihood_model_type=config["likelihood_model_type"],
            config_table_class=config_table_class,
            result_table_class=result_table_class,
            order_by_clause="test_post_pmf DESC",
            download_path="/tmp/",
        )
        config_id = row["config_id"]
        key["config_table_id"] = config_id
        elbo_model = torch.load(row["model"], map_location=self.device)
        posterior_model = elbo_model.posterior

        # load data
        print("Loading data")
        train_loader, validation_loader, test_loader = load_compiled_monkey_dataset(
            data_basepath=self.data_basepath,
            session_id=config["session_id"],
            image_crop=config["image_crop"],
            subsample=config["subsample"],
            scale=config["scale"],
            batch_size=config["batch_size"],
        )

        # compute metrics
        fill_dict_with_sysident_metrics(
            dict=key,
            config=config,
            model=posterior_model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            device=self.device,
        )

        # insert into table
        print("Inserting into table")
        self.insert1(key)


@schema
class ImageMarginalLikelihoodResult(dj.Computed):
    """
    Computes the marginal likelihood of images under the sampling generative model.
    Note that as config, PosteriorPMFApproxConfig is used since the generative model
    and n_samples is already specified there.
    """

    data_basepath = "/data/monkey/toliaslab/compiled_monkey_datasets/"
    device = "cpu"

    definition = """
    -> PosteriorPMFApproxConfig
    ---
    train_lpx_sum: float    # log p(x) using iwu, sum over pixels, mean over trials, in bits
    validation_lpx_sum: float    # log p(x) using iwu, sum over pixels, mean over trials, in bits
    test_lpx_sum: float    # log p(x) using iwu, sum over pixels, mean over trials, in bits

    config_table_id: char(32)   # id of the config table that stores the config of the trained elbo model.
    """

    def make(self, key):
        print("Populating ImageMarginalLikelihoodResult")
        print("Arguments in use:")
        print(f"Device: {self.device}")
        print(f"Data basepath: {self.data_basepath}")

        config = (PosteriorPMFApproxConfig & key).fetch1()

        config_table_class = config_table_class_map[config["config_table_name"]]
        result_table_class = result_table_class_map[config["result_table_name"]]
        print(f"Config table: {config_table_class.__name__}")
        print(f"Result table: {result_table_class.__name__}")

        # fetch best model for given config
        print("Fetching best model for given config")
        row = load_best_sampling_row(
            session_id=config["session_id"],
            prior_model_type=config["prior_model_type"],
            likelihood_model_type=config["likelihood_model_type"],
            config_table_class=config_table_class,
            result_table_class=result_table_class,
            order_by_clause="test_post_pmf DESC",
            download_path="/tmp/",
        )
        config_id = row["config_id"]
        key["config_table_id"] = config_id
        elbo_model = torch.load(row["model"], map_location=self.device)
        prior_model = elbo_model.joint.prior
        likelihood_model = elbo_model.joint.conditional

        # load data
        print("Loading data")
        train_loader, validation_loader, test_loader = load_compiled_monkey_dataset(
            data_basepath=self.data_basepath,
            session_id=config["session_id"],
            image_crop=config["image_crop"],
            subsample=config["subsample"],
            scale=config["scale"],
            batch_size=config["batch_size"],
        )

        print("Computing marginal likelihood")
        print("Train set")
        train_lpx_sum = marginal_likelihood(
            prior=prior_model,
            likelihood=likelihood_model,
            dataloader=train_loader,
            n_samples=config["n_samples"],
            device=self.device,
        )
        print("Validation set")
        validation_lpx_sum = marginal_likelihood(
            prior=prior_model,
            likelihood=likelihood_model,
            dataloader=validation_loader,
            n_samples=config["n_samples"],
            device=self.device,
        )
        print("Test set")
        test_lpx_sum = marginal_likelihood(
            prior=prior_model,
            likelihood=likelihood_model,
            dataloader=test_loader,
            n_samples=config["n_samples"],
            device=self.device,
        )

        key["train_lpx_sum"] = train_lpx_sum
        key["validation_lpx_sum"] = validation_lpx_sum
        key["test_lpx_sum"] = test_lpx_sum

        # insert into table
        print("Inserting into table")
        self.insert1(key)
