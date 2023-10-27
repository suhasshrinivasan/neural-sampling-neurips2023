from argparse import ArgumentParser as AP
from time import time

import numpy as np
import torch
from dequant_experiment_helpers import get_flow_dequantizer, get_flow_prior_model
from neural_sampling_code.dequantization.dequantization_factory import (
    DequantizerTrainer,
    get_dequantization_trainer,
)
from neural_sampling_code.elements.loss_functions import prior_mle
from neural_sampling_code.elements.trainers import Trainer
from neural_sampling_code.ml_tools.training_helpers import TrainLogger
from neural_sampling_code.utilities.utilities import (
    concat_responses,
    kurtosis,
    load_compiled_monkey_dataset,
    skew,
)
from torch import nn

import gensn.distributions as gd
from gensn.transforms.surjective import StepQuantizer
from gensn.variational import VariationalDequantizedDistribution


def evaluate_model(model, val_loader, sample_size=(10_000,)):
    all_val_responses = concat_responses(val_loader)
    true_response_mean = all_val_responses.mean(dim=0).detach().cpu().numpy()
    true_response_var = all_val_responses.var(dim=0).detach().cpu().numpy()
    true_kurtosis = kurtosis(all_val_responses).detach().cpu().numpy()
    true_skew = skew(all_val_responses).detach().cpu().numpy()

    model_responses = model.sample(sample_size)
    model_response_mean = model_responses.mean(dim=0).detach().cpu().numpy()
    model_response_var = model_responses.var(dim=0).detach().cpu().numpy()
    model_kurtosis = kurtosis(model_responses).detach().cpu().numpy()
    model_skew = skew(model_responses).detach().cpu().numpy()

    # compute the mse of true and model response mean, var, kurtosis, skew
    mse_response_mean = np.mean((true_response_mean - model_response_mean) ** 2)
    mse_response_var = np.mean((true_response_var - model_response_var) ** 2)
    mse_response_kurtosis = np.mean((true_kurtosis - model_kurtosis) ** 2)
    mse_response_skew = np.mean((true_skew - model_skew) ** 2)

    return (
        mse_response_mean,
        mse_response_var,
        mse_response_kurtosis,
        mse_response_skew,
    )


def dequant_prior_comparison_experiment(
    seed,
    session_id,
    image_crop,
    subsample,
    scale,
    batch_size,
    prior_type,
    device,
    n_epochs=100,
    lr=1e-3,
    data_basepath="/src/project/computed/compiled_monkey_datasets/",
    logging_type="wandb",
):
    # TODO: save configs in config
    # set random seed
    torch.manual_seed(seed=seed)
    rng = np.random.default_rng(seed=seed)

    # fetch data
    train_loader, val_loader, test_loader = load_compiled_monkey_dataset(
        data_basepath=data_basepath,
        session_id=session_id,
        image_crop=image_crop,
        subsample=subsample,
        scale=scale,
        batch_size=batch_size,
    )
    images, responses = next(iter(train_loader))
    n_neurons = responses.shape[1]

    # set up model

    # setup prior
    if prior_type == "flow":
        # NOTE: this is based on best results of flow prior attempts
        prior_config = dict(
            flow_depth=3,
            flow_initial_nonlinearity="inv_softplus",
            flow_nonlinearity="tanh",
            flow_base_distribution="normal",
        )
        prior_model = get_flow_prior_model(dims=n_neurons, **prior_config)
    elif prior_type == "exp":
        prior_model = gd.IndependentExponential(
            rate=nn.Parameter(torch.ones(n_neurons))
        )
    elif prior_type == "normal":
        prior_model = gd.IndependentNormal(
            loc=nn.Parameter(torch.zeros(n_neurons)),
            scale=nn.Parameter(torch.ones(n_neurons)),
        )
    elif prior_type == "laplace":
        prior_model = gd.IndependentLaplace(
            loc=nn.Parameter(torch.zeros(n_neurons)),
            scale=nn.Parameter(torch.ones(n_neurons)),
        )
    elif prior_type == "half_normal":
        prior_model = gd.IndependentHalfNormal(
            scale=nn.Parameter(torch.ones(n_neurons))
        )
    elif prior_type == "log_normal":
        prior_model = gd.IndependentLogNormal(
            loc=nn.Parameter(torch.zeros(n_neurons)),
            scale=nn.Parameter(torch.ones(n_neurons)),
        )
    elif prior_type == "exp_ones":
        prior_model = gd.IndependentExponential(rate=torch.ones(n_neurons))
    else:
        raise ValueError("Unknown prior type")

    # set up dequantizer
    # this config is based on best results of dequantization attempts
    dequantizer_config = dict(
        flow_depth=3,
        flow_initial_nonlinearity="inv_sigmoid",
        flow_nonlinearity="inv_elu",
        flow_base_distribution="conditional_normal",
    )

    dequantizer_model = get_flow_dequantizer(
        dims=n_neurons,
        **dequantizer_config,
    )

    # set up quantizer
    quantizer_model = StepQuantizer()

    # set up dequantization model
    model = VariationalDequantizedDistribution(
        prior=prior_model,
        dequantizer=dequantizer_model,
        quantizer=quantizer_model,
    )

    # setup trainer
    train_logger = TrainLogger(
        model_display_name="Dequantization",
        logging_type=logging_type,
    )
    trainer = DequantizerTrainer(
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        lr=lr,
        apply_early_stopping=True,
        early_stopping_patience=10,
        early_stopping_threshold=10,
        device=device,
        paired_data=True,
        logger=train_logger,
    )

    # train
    model, train_loss, val_loss = trainer.train_and_validate(
        model, train_loader, val_loader, n_epochs
    )

    # evaluate
    # this does evaluation in addition to validation log likelihood that
    # is already defined by the trainer
    # TODO: what about test set?
    (
        mse_response_mean,
        mse_response_var,
        mse_response_kurtosis,
        mse_response_skew,
    ) = evaluate_model(model, val_loader, sample_size=(10_000,))

    # log metrics
    final_metrics = dict(
        val_ll=-val_loss,
        train_ll=-train_loss,
        mse_response_mean=mse_response_mean,
        mse_response_var=mse_response_var,
        mse_response_kurtosis=mse_response_kurtosis,
        mse_response_skew=mse_response_skew,
    )

    train_logger.log(final_metrics, count_phrase="final metrics")

    return model, final_metrics
