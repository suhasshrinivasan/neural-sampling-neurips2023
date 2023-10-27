import torch
from database.schema.neurips2023.posterior_schema import (
    SampleBasedNNSysIdentResults2,
    SampleBasedSysIdentConfig2,
)
from database.schema.neurips2023.sysident_tables import (
    MonkeySysidentConfig3,
    MonkeySysidentResults3,
)
from gensn_experiments.monkey.prior.schema import DequantPriorExperimentResultLog

from nnsysident.models.models import Stacked2dPointPooled_Gamma
from nnvision.datasets.monkey_loaders import monkey_static_loader


def load_best_gamma_sysident_model(
    session_id,
    seed=42,
    image_crop=96,
    subsample=1,
    scale=1.0,
    time_bins_sum=12,
    batch_size=128,
    gamma_eps=1e-6,
    gamma_min_rate=0.0,
    gamma_max_conc=100,
    model_config=None,
    config_table_class=MonkeySysidentConfig3,
    result_table_class=MonkeySysidentResults3,
    download_path="/tmp/",
    order_by_clause="test_pmf DESC",
    model_class=Stacked2dPointPooled_Gamma,
):
    # setup dataset config
    dataset_config = {
        "dataset": "CSRF19_V1",
        "neuronal_data_files": [
            f"/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_{session_id}.pickle",
        ],
        "image_cache_path": "/data/monkey/toliaslab/CSRF19_V1/images/individual",
        "crop": image_crop,
        "subsample": subsample,
        "scale": scale,
        "seed": seed,
        "time_bins_sum": time_bins_sum,
        "batch_size": batch_size,
    }
    # setup model config
    if model_config is None:
        model_config = {
            "layers": 3,
            "input_kern": 24,
            "gamma_input": 10,
            "gamma_readout": 0.5,
            "hidden_dilation": 2,
            "hidden_kern": 9,
            "hidden_channels": 32,
        }

    # load dataloaders
    dataloaders = monkey_static_loader(**dataset_config)

    # build model
    model = model_class().build_model(
        dataloaders,
        seed,
        eps=gamma_eps,
        min_rate=gamma_min_rate,
        max_concentration=gamma_max_conc,
        **model_config,
    )

    # fetch dj result table
    table = config_table_class() * result_table_class()
    restriction = f"session_id = {session_id}"
    results = (table & restriction).fetch(
        order_by=order_by_clause,
        limit=1,
        as_dict=True,
        download_path=download_path,
    )
    if len(results) == 0:
        raise ValueError(
            f"No model with session_id {session_id} found in the database."
        )
    row = results[0]

    # load model state dict
    model_state_dict = torch.load(row["model_state_dict"])
    model.load_state_dict(model_state_dict)
    model = model.eval()
    return model


def load_best_sampling_row(
    session_id,
    prior_model_type,
    likelihood_model_type,
    config_table_class=SampleBasedSysIdentConfig2,
    result_table_class=SampleBasedNNSysIdentResults2,
    download_path="/tmp/",
    order_by_clause="test_post_pmf DESC",
):
    table = config_table_class() * result_table_class()
    restriction = f"session_id = {session_id} and prior_model_type = '{prior_model_type}' and likelihood_model_type = '{likelihood_model_type}'"
    results = (table & restriction).fetch(
        order_by=order_by_clause,
        limit=1,
        as_dict=True,
        download_path=download_path,
    )
    if len(results) == 0:
        raise ValueError(
            f"No model with session_id {session_id}, prior_model_type {prior_model_type} and likelihood_model_type {likelihood_model_type} found in the database."
        )

    row = results[0]
    return row


def load_best_sampling_model(
    session_id,
    prior_model_type,
    likelihood_model_type,
    config_table_class=SampleBasedSysIdentConfig2,
    result_table_class=SampleBasedNNSysIdentResults2,
    download_path="/tmp/",
    order_by_clause="test_post_pmf DESC",
):
    row = load_best_sampling_row(
        session_id,
        prior_model_type,
        likelihood_model_type,
        config_table_class=config_table_class,
        result_table_class=result_table_class,
        download_path=download_path,
        order_by_clause=order_by_clause,
    )
    model = torch.load(row["model"])
    model = model.eval()
    return model


def load_best_prior_model(
    session_id,
    prior_model_type,
    table_class=DequantPriorExperimentResultLog,
    download_path="/tmp/",
    order_by_clause="val_ll DESC",
):
    table = table_class()
    restriction = f"session_id = {session_id} and prior_type = '{prior_model_type}'"
    results = (table & restriction).fetch(
        order_by=order_by_clause,
        limit=1,
        as_dict=True,
        download_path=download_path,
    )
    if len(results) == 0:
        raise ValueError(
            f"No model with session_id {session_id} and prior_model_type {prior_model_type} found in the database."
        )

    row = results[0]
    model = torch.load(row["model"])
    model = model.eval()
    return model
