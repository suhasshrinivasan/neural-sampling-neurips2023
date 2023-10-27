from collections import namedtuple

import torch
from gensn_experiments.monkey.posterior.elbo_model_evaluation import (
    compute_elbo_metrics_on_dataloaders,
)
from neural_sampling_code.elements import loss_functions
from neural_sampling_code.elements.evaluation import (
    sysident_correlation,
    sysident_log_prob,
)
from neural_sampling_code.elements.loaders import ImageNamedTupleWrappedDataset
from neural_sampling_code.elements.trainers import Trainer
from neural_sampling_code.ml_tools.training_helpers import TrainLogger
from neural_sampling_code.utilities.utilities import load_compiled_monkey_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader

import gensn.distributions as G
import gensn.variational as V
from nnsysident.training.trainers import standard_trainer
from nnsysident.utility.measures import get_correlations, get_loss


def get_model_performance(
    model, dataloaders, loss_function, device="cpu", print_performance=True
):
    output = {"correlation": {}, "loss": {}}
    for tier in ["train", "validation", "test"]:
        output["correlation"][tier] = get_correlations(
            model, dataloaders[tier], device=device, per_neuron=False
        )

        output["loss"][tier] = get_loss(
            model,
            dataloaders[tier],
            loss_function,
            device=device,
            per_neuron=False,
            avg=True,
        )
    if print_performance:
        for measure, tiers in output.items():
            print("\u0332".join(measure + " "))
            print("")
            for tier, value in tiers.items():
                print(tier + ":" + " " * (13 - len(tier)) + "{0:.3f} ".format(value))
            print("")
    return output


def sample_based_posterior_experiment(
    seed,
    session_id,
    image_crop,
    subsample,
    scale,
    batch_size,
    prior_model,
    likelihood_model,
    posterior_model,
    l2_weight_decay,
    lr,
    n_epochs,
    apply_early_stopping,
    early_stopping_threshold,
    early_stopping_patience,
    apply_gradient_clipping,
    gradient_clipping_threshold,
    apply_update_skipping,
    update_skipping_threshold,
    device,
    sample_train_loader,
    sample_val_loader,
    sample_test_loader,
    early_stopping_criterion="val_loss",
    loss_type="posterior_mle",
    data_basepath="/data/",
    logging_type="stdout",
    n_eval_samples=100,
):
    """
    Function that trains a sample-based posterior model and evaluates it on the
    real as well as the sample data.

    Args:
        seed (int): random seed
        session_id (str): session id
        image_crop (int): image crop
        subsample (int): subsample
        scale (float): scale
        batch_size (int): batch size
        prior_model: trained prior model
        likelihood_model: trained likelihood model
        posterior_model: posterior model (build)
        l2_weight_decay (float): l2 weight decay
        lr (float): learning rate
        n_epochs (int): number of epochs
        apply_early_stopping (bool): whether to apply early stopping
        early_stopping_threshold (float): early stopping threshold
        early_stopping_patience (int): early stopping patience
        apply_gradient_clipping (bool): whether to apply gradient clipping
        gradient_clipping_threshold (float): gradient clipping threshold
        apply_update_skipping (bool): whether to apply update skipping
        update_skipping_threshold (float): update skipping threshold
        device (str): device
        sample_train_loader (torch.utils.data.DataLoader): sample train loader
        sample_val_loader (torch.utils.data.DataLoader): sample val loader
        sample_test_loader (torch.utils.data.DataLoader): sample test loader
        data_basepath (str): data basepath
        logging_type (str): logging type
    Returns:
        elbo_model (gensn.variational.ELBOMarginal): elbo_model
        metrics (dict): metrics
    """

    # set random seed
    torch.manual_seed(seed=seed)

    ################# set up elbo model #################
    # set up the joint model
    print("Setting up the joint model...")
    joint_model = G.Joint(prior_model, likelihood_model)

    # set up the elbo model
    elbo_model = V.ELBOMarginal(
        joint=joint_model,
        posterior=posterior_model,
    )

    ################# set up the loss function #################
    print("Setting up the loss function...")
    if loss_type == "posterior_mle":
        # since this is sample based posterior training, we only train the posterior model
        # while keeping the generative model fixed
        model_to_train = posterior_model
        # the loss function is negative log likelihood of posterior model given sample responses
        compute_loss = loss_functions.posterior_mle_with_uniform_noise_on_elbo
    elif loss_type == "elbo":
        model_to_train = posterior_model
        compute_loss = loss_functions.elbo
    else:
        raise ValueError(f"Unknown loss type {loss_type}!")

    if early_stopping_criterion == "val_loss":
        compute_early_stopping_loss = compute_loss
    else:
        raise ValueError(
            f"Unknown early stopping criterion {early_stopping_criterion}!"
        )
    # we could also default compute_early_stopping_loss to None, in which case
    # the trainer will automatically use compute_loss for early stopping loss
    # as long as apply_early_stopping is set to True

    ################# set up the trainer and train #################
    print("Setting up the trainer...")
    # set up the optimizer
    optimizer = Adam(
        params=model_to_train.parameters(),
        lr=lr,
        weight_decay=l2_weight_decay,
    )

    # set up the logger
    logger = TrainLogger(
        model_display_name="Sample-based-posterior",
        logging_type=logging_type,
    )
    # set up the trainer
    trainer = Trainer(
        optimizer=optimizer,
        lr=lr,
        logger=logger,
        device=device,
        compute_loss=compute_loss,
        apply_early_stopping=apply_early_stopping,
        compute_early_stopping_loss=compute_early_stopping_loss,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
        apply_gradient_clipping=apply_gradient_clipping,
        gradient_clipping_threshold=gradient_clipping_threshold,
        apply_update_skipping=apply_update_skipping,
        update_skipping_threshold=update_skipping_threshold,
    )

    print("Training...")
    # train
    # NOTE: training here needs to happen over samples instead of real data
    # hence pass in sample_train_loader and sample_val_loader
    # and not train_loader and val_loader!

    elbo_model, train_summary = trainer(
        elbo_model,
        sample_train_loader,
        sample_val_loader,
        n_epochs=n_epochs,
    )

    # omit last epoch, train_losses_batchwise, val_losses_batchwise and early_stopping_losses_batchwise
    # from train_summary since we don't need them
    _ = train_summary.pop("last_epoch")
    _ = train_summary.pop("train_losses_batchwise")
    _ = train_summary.pop("val_losses_batchwise")
    _ = train_summary.pop("early_stopping_losses_batchwise")

    ################# fetch real data for evaluation #################
    print("Fetching real data for evaluation...")
    train_loader, val_loader, test_loader = load_compiled_monkey_dataset(
        data_basepath=data_basepath,
        session_id=session_id,
        image_crop=image_crop,
        subsample=subsample,
        scale=scale,
        batch_size=batch_size,
    )

    ################# evaluate on sample as well as real dataloaders #################
    # all models at the moment are gamma based
    # TODO: make this more general
    posterior_distribution = "gamma"

    print("Evaluating on sample as well as real dataloaders...")
    all_metrics = compute_elbo_metrics_on_dataloaders(
        elbo_model=elbo_model,
        device=device,
        posterior_distribution=posterior_distribution,
        n_samples=n_eval_samples,
        **{
            "sample_train": sample_train_loader,
            "sample_val": sample_val_loader,
            "sample_test": sample_test_loader,
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        },
    )

    # log
    print("Logging...")
    logger.log(metrics=all_metrics, count_phrase="Final evaluation metrics")

    # prepare results to return
    final_results = {
        **all_metrics,
        **train_summary,
    }

    print("Returning results...")
    return elbo_model, final_results


def sample_based_nnsysident_experiment(
    seed,
    session_id,
    image_crop,
    subsample,
    scale,
    batch_size,
    prior_model,
    likelihood_model,
    posterior_model,
    trainer_config,
    loss_function,
    device,
    sample_train_loader,
    sample_val_loader,
    sample_test_loader,
    data_basepath="/data/",
    n_eval_samples=100,
):
    """
    Function that trains a sample-based nnsysident model using nnsysident trainer
    and evaluates it on the real as well as the sample data.

    Args:
        seed (int): random seed
        session_id (str): session id
        image_crop (int): image crop
        subsample (int): subsample
        scale (float): scale
        batch_size (int): batch size
        prior_model: trained prior model
        likelihood_model: trained likelihood model
        posterior_model: posterior model (build)
        trainer_config (dict): trainer config
        loss_function (str): loss function, example "GammaLoss"
        device (str): device
        sample_train_loader (torch.utils.data.DataLoader): sample train loader
        sample_val_loader (torch.utils.data.DataLoader): sample val loader
        sample_test_loader (torch.utils.data.DataLoader): sample test loader
        data_basepath (str): data basepath
    Returns:
        elbo_model (gensn.variational.ELBOMarginal): elbo_model
        metrics (dict): metrics
    """

    # set random seed
    torch.manual_seed(seed=seed)

    ################# set up elbo model #################
    # set up the joint model
    print("Setting up the joint model...")
    joint_model = G.Joint(prior_model, likelihood_model)

    # set up the elbo model
    elbo_model = V.ELBOMarginal(
        joint=joint_model,
        posterior=posterior_model,
    )

    ################# prepare the dataloaders for nnsysident training #################
    # nnsysident trainer expects .dataset to consist of namedtuples with fields
    # inputs and targets where inputs is images and targets is neuronal responses
    print("Preparing the dataloaders for nnsysident training...")
    # NOTE: training here needs to happen over samples instead of real data
    # hence pass in sample_train_loader and sample_val_loader
    # and not train_loader and val_loader!
    DataPoint = namedtuple("DataPoint", ["inputs", "targets"])
    sample_nnsysident_train_loader = DataLoader(
        ImageNamedTupleWrappedDataset(DataPoint, sample_train_loader.dataset),
        batch_size=batch_size,
        shuffle=True,
    )
    sample_nnsysident_val_loader = DataLoader(
        ImageNamedTupleWrappedDataset(DataPoint, sample_val_loader.dataset),
        batch_size=batch_size,
        shuffle=False,
    )
    sample_nnsysident_test_loader = DataLoader(
        ImageNamedTupleWrappedDataset(DataPoint, sample_test_loader.dataset),
        batch_size=batch_size,
        shuffle=False,
    )
    # Furthermore, nnsysident trainer expects teh dataloaders to a dict of dicts
    sample_dataloaders = {
        "train": {
            session_id: sample_nnsysident_train_loader,
        },
        "validation": {
            session_id: sample_nnsysident_val_loader,
        },
        "test": {
            session_id: sample_nnsysident_test_loader,
        },
    }

    ################# fetch real data for evaluation #################
    print("Fetching real data for evaluation...")
    real_train_loader, real_val_loader, real_test_loader = load_compiled_monkey_dataset(
        data_basepath=data_basepath,
        session_id=session_id,
        image_crop=image_crop,
        subsample=subsample,
        scale=scale,
        batch_size=batch_size,
    )

    # Prepare also the real data dataloaders for evaluation
    real_nnsysident_train_loader = DataLoader(
        ImageNamedTupleWrappedDataset(DataPoint, real_train_loader.dataset),
        batch_size=batch_size,
        shuffle=True,
    )
    real_nnsysident_val_loader = DataLoader(
        ImageNamedTupleWrappedDataset(DataPoint, real_val_loader.dataset),
        batch_size=batch_size,
        shuffle=False,
    )
    real_nnsysident_test_loader = DataLoader(
        ImageNamedTupleWrappedDataset(DataPoint, real_test_loader.dataset),
        batch_size=batch_size,
        shuffle=False,
    )

    # Furthermore, nnsysident trainer expects teh dataloaders to a dict of dicts
    real_dataloaders = {
        "train": {
            session_id: real_nnsysident_train_loader,
        },
        "validation": {
            session_id: real_nnsysident_val_loader,
        },
        "test": {
            session_id: real_nnsysident_test_loader,
        },
    }

    ################# set the model to pass to nnsysident trainer #################
    sysident_model = posterior_model.trainable_distribution.parameter_generator[1]

    ################# set up the trainer and train #################
    print("Training...")
    # # train
    # # NOTE: training here needs to happen over samples instead of real data
    # # hence pass in sample_train_loader and sample_val_loader
    # # and not train_loader and val_loader!
    gamma_score, gamma_output, gamma_state_dict = standard_trainer(
        sysident_model,
        sample_dataloaders,
        seed,
        loss_function=loss_function,
        track_training=True,
        **trainer_config,
    )

    gamma_performance_on_sample = get_model_performance(
        sysident_model, sample_dataloaders, loss_function=loss_function, device=device
    )
    gamma_correlation_on_sample = gamma_performance_on_sample["correlation"]
    # add "sample" to the keys
    gamma_correlation_on_sample = {
        "sample_" + key + "_correlation": value
        for key, value in gamma_correlation_on_sample.items()
    }

    gamma_performance_on_real = get_model_performance(
        sysident_model, real_dataloaders, loss_function=loss_function, device=device
    )
    gamma_correlation_on_real = gamma_performance_on_real["correlation"]
    # add "sample" to the keys
    gamma_correlation_on_real = {
        key + "_correlation": value for key, value in gamma_correlation_on_real.items()
    }

    ################# evaluate on sample as well as real dataloaders #################
    # all models at the moment are gamma based
    # TODO: make this more general
    posterior_distribution = "gamma"

    print("Evaluating on sample as well as real dataloaders...")
    all_metrics = compute_elbo_metrics_on_dataloaders(
        elbo_model=elbo_model,
        device=device,
        posterior_distribution=posterior_distribution,
        n_samples=n_eval_samples,
        **{
            "sample_train": sample_train_loader,
            "sample_val": sample_val_loader,
            "sample_test": sample_test_loader,
            "train": real_train_loader,
            "val": real_val_loader,
            "test": real_test_loader,
        },
    )

    # remove all correlations from all_metrics since we already have them
    # in gamma_correlation_on_sample and gamma_correlation_on_real
    all_metrics = {
        key: value
        for key, value in all_metrics.items()
        if not key.endswith("correlation")
    }

    # log
    print("Logging...")
    print(all_metrics)

    # prepare results to return
    final_results = {
        **all_metrics,
        **gamma_output,
        **gamma_correlation_on_sample,
        **gamma_correlation_on_real,
    }

    print("Returning results...")
    return elbo_model, final_results


def get_sysident_performance(model, dataloaders, session_id, distribution_type, device):
    correlation_function_lookup = {
        "poisson": sysident_correlation,
    }
    correlation_function = correlation_function_lookup[distribution_type]
    output = {}
    for tier in ["train", "validation", "test"]:
        pmf_sum, pmf_mean, pmf_sem = sysident_log_prob(
            model,
            dataloaders[tier][session_id],
            device=device,
        )
        correlation, correlation_sem = correlation_function(
            model,
            dataloaders[tier][session_id],
            device=device,
        )
        output.update(
            {
                tier + "_pmf_sum": pmf_sum,
                tier + "_pmf_mean": pmf_mean,
                tier + "_pmf_sem": pmf_sem,
                tier + "_correlation": correlation,
                tier + "_correlation_sem": correlation_sem,
            }
        )
    return output


def sample_based_nnsysident_general_posterior_experiment(
    seed,
    session_id,
    image_crop,
    subsample,
    scale,
    batch_size,
    prior_model,
    likelihood_model,
    posterior_distribution,
    posterior_model,
    trainer_config,
    loss_function,
    device,
    sample_train_loader,
    sample_val_loader,
    sample_test_loader,
    data_basepath="/data/monkey/toliaslab/compiled_monkey_datasets/",
):
    """
    Function that trains a sample-based nnsysident model using nnsysident trainer
    and evaluates it on the real as well as the sample data.

    At its core this function is the same as sample_based_nnsysident_experiment
    except that it uses a general posterior model instead of a gamma posterior model, and
    only returns sysident metrics on real as well as sample data.

    Args:
        seed (int): random seed
        session_id (str): session id
        image_crop (int): image crop
        subsample (int): subsample
        scale (float): scale
        batch_size (int): batch size
        prior_model: trained prior model
        likelihood_model: trained likelihood model
        posterior_model: posterior model (build)
        trainer_config (dict): trainer config
        loss_function (str): loss function, example "GammaLoss"
        device (str): device
        sample_train_loader (torch.utils.data.DataLoader): sample train loader
        sample_val_loader (torch.utils.data.DataLoader): sample val loader
        sample_test_loader (torch.utils.data.DataLoader): sample test loader
        data_basepath (str): data basepath
    Returns:
        elbo_model (gensn.variational.ELBOMarginal): elbo_model
        metrics (dict): metrics
    """

    # set random seed
    torch.manual_seed(seed=seed)

    ################# set up elbo model #################
    # set up the joint model
    print("Setting up the joint model...")
    joint_model = G.Joint(prior_model, likelihood_model)

    # set up the elbo model
    elbo_model = V.ELBOMarginal(
        joint=joint_model,
        posterior=posterior_model,
    )

    ################# prepare the dataloaders for nnsysident training #################
    # nnsysident trainer expects .dataset to consist of namedtuples with fields
    # inputs and targets where inputs is images and targets is neuronal responses
    print("Preparing the dataloaders for nnsysident training...")
    # NOTE: training here needs to happen over samples instead of real data
    # hence pass in sample_train_loader and sample_val_loader
    # and not train_loader and val_loader!
    DataPoint = namedtuple("DataPoint", ["inputs", "targets"])
    sample_nnsysident_train_loader = DataLoader(
        ImageNamedTupleWrappedDataset(DataPoint, sample_train_loader.dataset),
        batch_size=batch_size,
        shuffle=True,
    )
    sample_nnsysident_val_loader = DataLoader(
        ImageNamedTupleWrappedDataset(DataPoint, sample_val_loader.dataset),
        batch_size=batch_size,
        shuffle=False,
    )
    sample_nnsysident_test_loader = DataLoader(
        ImageNamedTupleWrappedDataset(DataPoint, sample_test_loader.dataset),
        batch_size=batch_size,
        shuffle=False,
    )
    # Furthermore, nnsysident trainer expects the dataloaders to a dict of dicts
    sample_dataloaders = {
        "train": {
            session_id: sample_nnsysident_train_loader,
        },
        "validation": {
            session_id: sample_nnsysident_val_loader,
        },
        "test": {
            session_id: sample_nnsysident_test_loader,
        },
    }

    ################# fetch real data for evaluation #################
    print("Fetching real data for evaluation...")
    real_train_loader, real_val_loader, real_test_loader = load_compiled_monkey_dataset(
        data_basepath=data_basepath,
        session_id=session_id,
        image_crop=image_crop,
        subsample=subsample,
        scale=scale,
        batch_size=batch_size,
    )

    # Prepare also the real data dataloaders for evaluation
    real_nnsysident_train_loader = DataLoader(
        ImageNamedTupleWrappedDataset(DataPoint, real_train_loader.dataset),
        batch_size=batch_size,
        shuffle=True,
    )
    real_nnsysident_val_loader = DataLoader(
        ImageNamedTupleWrappedDataset(DataPoint, real_val_loader.dataset),
        batch_size=batch_size,
        shuffle=False,
    )
    real_nnsysident_test_loader = DataLoader(
        ImageNamedTupleWrappedDataset(DataPoint, real_test_loader.dataset),
        batch_size=batch_size,
        shuffle=False,
    )

    # Furthermore, nnsysident trainer expects teh dataloaders to a dict of dicts
    real_dataloaders = {
        "train": {
            session_id: real_nnsysident_train_loader,
        },
        "validation": {
            session_id: real_nnsysident_val_loader,
        },
        "test": {
            session_id: real_nnsysident_test_loader,
        },
    }

    ################# set the model to pass to nnsysident trainer #################
    sysident_model = posterior_model.trainable_distribution.parameter_generator[1]

    ################# set up the trainer and train #################
    print("Training...")
    # # train
    # # NOTE: training here needs to happen over samples instead of real data
    # # hence pass in sample_train_loader and sample_val_loader
    # # and not train_loader and val_loader!
    score, output, state_dict = standard_trainer(
        sysident_model,
        sample_dataloaders,
        seed,
        loss_function=loss_function,
        track_training=True,
        **trainer_config,
    )

    print("Evaluating on sample dataloaders")

    performance_on_sample = get_sysident_performance(
        posterior_model,
        sample_dataloaders,
        session_id,
        posterior_distribution,
        device="cpu",
    )
    # add "sample" to the keys
    performance_on_sample = {
        "sample_" + key: value for key, value in performance_on_sample.items()
    }

    print("Evaluating on real dataloaders")
    performance_on_real = get_sysident_performance(
        posterior_model,
        real_dataloaders,
        session_id,
        posterior_distribution,
        device="cpu",
    )
    performance = {
        **performance_on_sample,
        **performance_on_real,
    }
    print(performance)
    print("returning")
    return elbo_model, performance
