from pathlib import Path
from time import time

import numpy as np
import torch
from neural_sampling_code.elements.conditioned_parameters import (
    MeanStdMLPExp,
    MeanStdMLPFlexible,
)
from neural_sampling_code.elements.loss_functions import likelihood_mle
from neural_sampling_code.elements.trainers import Trainer
from neural_sampling_code.flexible_models.trainers import GensnDecoderTrainer
from neural_sampling_code.ml_tools.training_helpers import TrainLogger
from neural_sampling_code.utilities.utilities import load_compiled_monkey_dataset
from numpy.random import default_rng
from torch.optim import Adam

import gensn.distributions as G
import wandb
from gensn.distributions import TrainableDistributionAdapter


def likelihood_experiment(
    seed,
    session_id,
    image_crop,
    subsample,
    scale,
    batch_size,
    n_layers,
    nonlinearity,
    dropout_rate,
    nonneg_std_transform,
    init_std,
    l2_weight_decay,
    lr,
    n_epochs,
    early_stopping_threshold,
    early_stopping_patience,
    gradient_clipping_threshold,
    device,
    data_basepath="/data/",
    logging_type="stdout",
):
    # set random seed
    torch.manual_seed(seed=seed)
    rng = np.random.default_rng(seed=seed)

    # fetch data and record dims
    train_loader, val_loader, test_loader = load_compiled_monkey_dataset(
        data_basepath=data_basepath,
        session_id=session_id,
        image_crop=image_crop,
        subsample=subsample,
        scale=scale,
        batch_size=batch_size,
    )
    images, responses = next(iter(train_loader))
    image_dim = images.flatten(start_dim=1).shape[-1]
    n_neurons = responses.shape[1]

    # set up decoder model
    decoder_param_model = MeanStdMLPFlexible(
        in_features=n_neurons,
        out_features=image_dim,
        n_layers=n_layers,
        nonlinearity=nonlinearity,
        dropout_rate=dropout_rate,
        nonneg_std_transform=nonneg_std_transform,
        init_std=init_std,
    )

    model = G.IndependentNormal(_parameters=decoder_param_model)

    # setup logger
    train_logger = TrainLogger(
        model_display_name="Likelihood",
        logging_type=logging_type,
    )

    # set up optimizer
    optimizer = Adam(
        decoder_param_model.parameters(),
        lr=lr,
        weight_decay=l2_weight_decay,
    )

    # set up trainer
    # gradient clipping is applied by default
    # early stopping is applied by default
    # update skipping is not applied
    trainer = Trainer(
        optimizer=optimizer,
        lr=lr,
        logger=train_logger,
        device=device,
        compute_loss=likelihood_mle,
        apply_early_stopping=True,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
        apply_gradient_clipping=True,
        gradient_clipping_threshold=gradient_clipping_threshold,
        apply_update_skipping=False,
        update_skipping_threshold=None,
    )

    # train model
    model, train_summary = trainer(
        model,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
    )

    # collect metrics
    metrics = {
        "train_ll": -train_summary["best_model_train_loss"],
        "val_ll": -train_summary["best_model_val_loss"],
    }

    # log metrics
    train_logger.log(metrics, count_phrase="final metrics")

    return model, metrics
