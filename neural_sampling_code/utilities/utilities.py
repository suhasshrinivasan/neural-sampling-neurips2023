import hashlib
import itertools
import json
import os
import pickle
from collections import Iterable, Mapping, OrderedDict
from hashlib import md5
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from neuralpredictors.measures import corr
from nnfabrik import builder
from nnsysident.datasets.mouse_loaders import (
    static_loaders as nnsysident_static_loaders,
)
from pandas import DataFrame as DF
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset


def serialize(obj):
    return pickle.dumps(obj)


def get_hash(obj):
    return md5(serialize(obj)).hexdigest()


def get_combinations(*iterables, hash=True):
    if hash:
        combinations = [
            (get_hash(prod), *prod) for prod in itertools.product(*iterables)
        ]
    else:
        combinations = itertools.product(*iterables)
    return combinations


# from nnfabrik:
# https://github.com/sinzlab/nnfabrik/blob/ea4f5148c943741e45d937fe7ee681978b4224f7/nnfabrik/utility/dj_helpers.py#L58-L97
def make_hash(obj):
    """
    Given a Python object, returns a 32 character hash string to uniquely identify
    the content of the object. The object can be arbitrary nested (i.e. dictionary
    of dictionary of list etc), and hashing is applied recursively to uniquely
    identify the content.
    For dictionaries (at any level), the key order is ignored when hashing
    so that {"a":5, "b": 3, "c": 4} and {"b": 3, "a": 5, "c": 4} will both
    give rise to the same hash. Exception to this rule is when an OrderedDict
    is passed, in which case difference in key order is respected. To keep
    compatible with previous versions of Python and the assumed general
    intentions, key order will be ignored even in Python 3.7+ where the
    default dictionary is officially an ordered dictionary.
    Args:
        obj - A (potentially nested) Python object
    Returns:
        hash: str - a 32 charcter long hash string to uniquely identify the object.
    """
    hashed = hashlib.md5()

    if isinstance(obj, str):
        hashed.update(obj.encode())
    elif isinstance(obj, OrderedDict):
        for k, v in obj.items():
            hashed.update(str(k).encode())
            hashed.update(make_hash(v).encode())
    elif isinstance(obj, Mapping):
        for k in sorted(obj, key=str):
            hashed.update(str(k).encode())
            hashed.update(make_hash(obj[k]).encode())
    elif isinstance(obj, Iterable):
        for v in obj:
            hashed.update(make_hash(v).encode())
    else:
        hashed.update(str(obj).encode())

    return hashed.hexdigest()


def hash_args(f):
    def wrapper(*args, **kwargs):
        args_hash = make_hash((args, kwargs))
        return f(*args, **kwargs, _hash=args_hash)

    return wrapper


def concat_responses(dataloader):
    return torch.cat([responses for _, responses in dataloader], dim=0)


def evaluate_on_true_responses(dataloader, eval_fn):
    true_evals = []
    for _, responses in dataloader:
        true_evals.append(eval_fn(responses))

    true_evals = torch.cat(true_evals, dim=0)
    return true_evals


# TODO: also think of posterior responses
def evaluate_on_model_responses(eval_fn, model, device, sample_size=torch.Size([])):
    model_response = model.sample(sample_size)
    return eval_fn(model_response)


def eval_fn(responses, metric="mse"):
    if metric == "mse":
        return torch.mean((responses - responses.mean(dim=0)) ** 2, dim=0)
    elif metric == "var":
        return torch.var(responses, dim=0)
    elif metric == "mean":
        return torch.mean(responses, dim=0)
    else:
        raise ValueError("Unknown metric")


# compute kurtosis
def kurtosis(responses):
    return (
        torch.mean((responses - responses.mean(dim=0)) ** 4, dim=0)
        / torch.var(responses, dim=0) ** 2
    )


# compute skew
def skew(responses):
    return (
        torch.mean((responses - responses.mean(dim=0)) ** 3, dim=0)
        / torch.var(responses, dim=0) ** 1.5
    )


def plot_distributions(*dists, dist_names, n_samples=1000):
    fig, ax = plt.subplots()
    fig.set_dpi(150)
    for dist, dist_name in zip(dists, dist_names):
        z = dist.sample((n_samples,))
        ax.scatter(z[:, 0], z[:, 1], s=5, label=dist_name)
    ax.set_aspect("equal")
    ax.legend()
    sns.despine(fig=fig, ax=ax, trim=True)


def plot_samples(samples, dist_names=None):
    fig, ax = plt.subplots()
    fig.set_dpi(150)
    if dist_names is None:
        dist_names = [None] * len(samples)
    for sample, dist_name in zip(samples, dist_names):
        ax.scatter(sample[:, 0], sample[:, 1], s=5, label=dist_name)
    ax.set_aspect("equal")
    ax.legend()
    sns.despine(fig=fig, ax=ax, trim=True)


def log_log_likelihood(log_pxr, log_px_r, log_pr):
    print("Printing log lik")
    print(f"log p(x, r) = {log_pxr: e}")
    print(f"log p(x|r) = {log_px_r: e}")
    print(f"log p(r) = {log_pr: e}")


def compare_likelihoods(
    log_pxr_train,
    log_pxr_test,
    log_px_r_train,
    log_px_r_test,
    log_pr_train,
    log_pr_test,
):
    print("Comparing train and test log likelihoods")
    print(f"Delta log p(x, r) = {log_pxr_train - log_pxr_test: e}")
    print(f"Delta log p(x|r) = {log_px_r_train - log_px_r_test: e}")
    print(f"Delta log p(r) = {log_pr_train - log_pr_test: e}")


def make_matched_neurons_dataset_config(sessions, device="cuda"):
    prefix = "/data/mouse/toliaslab/static/"
    multi_id_suffix = "meta/neurons/multi_match_id.npy"
    unit_id_suffix = "meta/neurons/unit_ids.npy"
    dirnames = [prefix + session for session in sessions]
    multi_id_filenames = [prefix + session + multi_id_suffix for session in sessions]
    unit_id_filenames = [prefix + session + unit_id_suffix for session in sessions]
    multi_ids = [np.load(filename) for filename in multi_id_filenames]
    unit_ids = [np.load(filename) for filename in unit_id_filenames]

    common_multi_ids = list(set.intersection(*map(set, multi_ids)))
    common_unit_ids = [
        unit_id[np.isin(common_multi_ids, multi_id).nonzero()[0]]
        for multi_id, unit_id in zip(multi_ids, unit_ids)
    ]

    cuda = True if device == "cuda" else False
    dataset_config = {
        "paths": dirnames,
        "normalize": True,
        "neuron_ids": common_unit_ids,
        "include_behavior": False,
        "include_eye_position": False,
        "batch_size": 128,
        "scale": 1,
        "cuda": cuda,
    }
    dataset_fn = "sensorium.datasets.static_loaders"

    return dataset_config, dataset_fn


def load_matched_neurons_dataset(sessions, device="cuda"):
    # typically the sessions that we use are
    # sessions = [
    # "static22564-3-12-preproc0/", "static22564-3-8-preproc0/", "static22564-2-13-preproc0/", "static22564-2-12-preproc0/"
    # ]
    dataset_config, dataset_fn = make_matched_neurons_dataset_config(sessions, device)
    dataloaders = builder.get_data(dataset_fn, dataset_config)
    concat_train_dataloaders = DataLoader(
        ConcatDataset(
            [dataloader.dataset for dataloader in dataloaders["train"].values()]
        ),
        batch_size=128,
    )
    concat_validation_dataloaders = DataLoader(
        ConcatDataset(
            [dataloader.dataset for dataloader in dataloaders["validation"].values()]
        ),
        batch_size=128,
    )
    concat_test_dataloaders = DataLoader(
        ConcatDataset(
            [dataloader.dataset for dataloader in dataloaders["test"].values()]
        ),
        batch_size=128,
    )
    return (
        concat_train_dataloaders,
        concat_validation_dataloaders,
        concat_test_dataloaders,
    )


def load_session_matched_neurons_dataset(sessions, dataset_config):
    prefix = "/data/mouse/toliaslab/static/"
    multi_id_suffix = "meta/neurons/multi_match_id.npy"
    unit_id_suffix = "meta/neurons/unit_ids.npy"

    dirnames = [prefix + session for session in sessions]
    multi_id_filenames = [prefix + session + multi_id_suffix for session in sessions]
    unit_id_filenames = [prefix + session + unit_id_suffix for session in sessions]
    multi_ids = [np.load(filename) for filename in multi_id_filenames]
    unit_ids = [np.load(filename) for filename in unit_id_filenames]

    common_multi_ids = list(set.intersection(*map(set, multi_ids)))
    common_multi_ids = [
        common_multi_id for common_multi_id in common_multi_ids if common_multi_id != -1
    ]

    unit_id_indices = [
        [np.where(multi_id == cmi)[0][0] for cmi in common_multi_ids]
        for multi_id in multi_ids
    ]
    common_unit_ids = [
        unit_id[unit_id_idx] for unit_id, unit_id_idx in zip(unit_ids, unit_id_indices)
    ]

    dataset_config["neuron_ids"] = common_unit_ids
    dataset_config["paths"] = dirnames

    # On-the-fly import
    from nnsysident.datasets.mouse_loaders import (
        static_loaders as nnsysident_static_loaders,
    )

    dataloaders = nnsysident_static_loaders(**dataset_config)

    train_loader = dataloaders["train"]
    train_images = []
    train_responses = []
    for key in train_loader.keys():
        for img, resp in train_loader[key]:
            train_images.append(img)
            train_responses.append(resp)
    train_images = torch.cat(train_images)
    train_responses = torch.cat(train_responses)

    combined_train_loader = DataLoader(
        TensorDataset(train_images, train_responses), batch_size=128
    )

    validation_loader = dataloaders["validation"]
    validation_images = []
    validation_responses = []
    for key in validation_loader.keys():
        for img, resp in validation_loader[key]:
            validation_images.append(img)
            validation_responses.append(resp)
    validation_images = torch.cat(validation_images)
    validation_responses = torch.cat(validation_responses)

    combined_validation_loader = DataLoader(
        TensorDataset(validation_images, validation_responses), batch_size=128
    )

    test_loader = dataloaders["test"]
    test_images = []
    test_responses = []
    for key in test_loader.keys():
        for img, resp in test_loader[key]:
            test_images.append(img)
            test_responses.append(resp)
    test_images = torch.cat(test_images)
    test_responses = torch.cat(test_responses)

    combined_test_loader = DataLoader(
        TensorDataset(test_images, test_responses), batch_size=128
    )

    return (
        combined_train_loader,
        combined_validation_loader,
        combined_test_loader,
    )


def KL_between_1dgaussians(mu_1, logvar_1, mu_2, logvar_2):
    return -torch.sum(
        0.5
        * (
            1
            + logvar_1
            - logvar_2
            - logvar_1.exp() / logvar_2.exp()
            + ((mu_1 - mu_2) ** 2) / logvar_2.exp()
        )
    )


def print_and_log(log_dict):
    pprint_dict(log_dict)
    wandb.log(log_dict)


def pprint_dict(d):
    serialized_d = {
        k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in d.items()
    }
    print(json.dumps(serialized_d, indent=4, sort_keys=True))


def get_sweep_data(entity, project, sweep_id):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    sweep_runs = sweep.runs
    return DF([{**run.config, **(run.summary._json_dict)} for run in sweep_runs])


def get_monkey_data(
    data_basepath, dataset_fn, image_crop=96, subsample=1, scale=1, seed=1000
):
    """
    Load the data using the given basepath and dataset_fn.
    """
    neuronal_data_path = os.path.join(data_basepath, "neuronal_data/")
    neuronal_data_files = [
        neuronal_data_path + f
        for f in listdir(neuronal_data_path)
        if isfile(join(neuronal_data_path, f))
    ]
    image_cache_path = os.path.join(data_basepath, "images/individual")

    dataset_config = dict(
        dataset="CSRF19_V1",
        neuronal_data_files=neuronal_data_files,
        image_cache_path=image_cache_path,
        subsample=subsample,
        seed=seed,
        crop=image_crop,
        scale=scale,
    )
    return builder.get_data(dataset_fn, dataset_config)


def get_single_session_monkey_data(
    data_basepath,
    dataset_fn,
    session,
    image_crop=96,
    subsample=1,
    scale=1.0,
    batch_size=128,
):
    """
    Load the data using the given basepath and dataset_fn.
    """
    neuronal_data_path = os.path.join(data_basepath, "neuronal_data/")
    neuronal_data_files = [
        neuronal_data_path + f
        for f in listdir(neuronal_data_path)
        if isfile(join(neuronal_data_path, f))
    ]
    image_cache_path = os.path.join(data_basepath, "images/individual")

    dataset_config = dict(
        dataset="CSRF19_V1",
        neuronal_data_files=neuronal_data_files,
        image_cache_path=image_cache_path,
        batch_size=batch_size,
        subsample=subsample,
        seed=1000,
        crop=image_crop,
        scale=scale,
    )
    dataloaders_all_sessions = builder.get_data(dataset_fn, dataset_config)
    return {
        "train": dataloaders_all_sessions["train"][session],
        "validation": dataloaders_all_sessions["validation"][session],
        "test": dataloaders_all_sessions["test"][session],
    }


def extract_single_session_image_response(dataloader):
    return (
        dataloader.dataset[:].inputs.squeeze(1).flatten(start_dim=1),
        dataloader.dataset[:].targets,
    )


# Loads the monkey images and responses and constructs the dataloader
# for the given session, image_crop, subsample, scale, and batch_size.
# This function was implemented in order to load data from a "compiled"
# set of images and responses, which are stored in .pt files, such that
# the data loading is much faster on a cluster like HYAK
# than the original data loading function, which collects images and responses
# from a large number of sequential npy file reads.
def load_compiled_monkey_dataset(
    data_basepath, session_id, image_crop, subsample, scale, batch_size
):
    train_images_fname = (
        data_basepath + f"train_images_{session_id}_{image_crop}_{subsample}_{scale}.pt"
    )
    train_responses_fname = data_basepath + f"train_responses_{session_id}.pt"
    train_images = torch.load(train_images_fname)
    train_responses = torch.load(train_responses_fname)
    train_loader = DataLoader(
        TensorDataset(train_images, train_responses),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_images_fname = (
        data_basepath
        + f"validation_images_{session_id}_{image_crop}_{subsample}_{scale}.pt"
    )
    validation_responses_fname = data_basepath + f"validation_responses_{session_id}.pt"
    validation_images = torch.load(validation_images_fname)
    validation_responses = torch.load(validation_responses_fname)
    validation_loader = DataLoader(
        TensorDataset(validation_images, validation_responses),
        batch_size=batch_size,
    )
    test_images_fname = (
        data_basepath + f"test_images_{session_id}_{image_crop}_{subsample}_{scale}.pt"
    )
    test_responses_fname = data_basepath + f"test_responses_{session_id}.pt"
    test_images = torch.load(test_images_fname)
    test_responses = torch.load(test_responses_fname)
    test_loader = DataLoader(
        TensorDataset(test_images, test_responses), batch_size=batch_size
    )
    return train_loader, validation_loader, test_loader


def logmeanexp(x):
    return torch.logsumexp(x) - np.log(x.shape[0])


def get_compiled_data_from_config(
    config, DATA_PATH="/data/monkey/toliaslab/compiled_monkey_datasets/"
):
    train_loader_fname = (
        DATA_PATH
        + f"train_{config['session_id']}_{config['image_crop']}_{config['subsample']}_{config['scale']}.pt"
    )
    val_loader_fname = (
        DATA_PATH
        + f"validation_{config['session_id']}_{config['image_crop']}_{config['subsample']}_{config['scale']}.pt"
    )
    test_loader_fname = (
        DATA_PATH
        + f"test_{config['session_id']}_{config['image_crop']}_{config['subsample']}_{config['scale']}.pt"
    )
    return (
        torch.load(train_loader_fname),
        torch.load(val_loader_fname),
        torch.load(test_loader_fname),
    )


def preprocess_mouse_response_tensor(
    response, zero_threshold=torch.exp(torch.tensor(-10))
):
    response[response < 0] = zero_threshold / 2
    return response


def plottable_tensor_1d(tensor):
    return tensor.detach().cpu().numpy().flatten()


def get_gamma_model_mean(independent_gamma_dist):
    return (
        independent_gamma_dist.base_dist.concentration.squeeze(0).detach().cpu().numpy()
        / independent_gamma_dist.base_dist.rate.squeeze(0).detach().cpu().numpy()
    )


def get_poisson_model_mean(independent_poisson_dist):
    return independent_poisson_dist.base_dist.rate.squeeze(0).detach().cpu().numpy()


def average_correlation(models, test_dataloader, model_dist_type="gamma"):
    device = next(models[0].parameters()).device
    test_response_means = []
    all_model_means = []
    for images, responses in test_dataloader:
        images = images.to(device)
        responses = responses.to(device)
        response_mean = responses.mean(dim=0).cpu().detach().numpy()
        test_response_means.append(response_mean)
        model_means = []
        for model in models:
            model.eval()
            with torch.no_grad():
                if model_dist_type == "gamma" or model_dist_type == "independent_gamma":
                    model_dist = model(
                        images[0]
                    )  # images are repeats of the same image
                    model_mean = get_gamma_model_mean(model_dist)
                    model_means.append(model_mean)
                elif (
                    model_dist_type == "poisson"
                    or model_dist_type == "independent_poisson"
                ):
                    model_dist = model(images[0])
                    model_mean = model_dist.mean.detach().cpu().numpy()
                    model_means.append(model_mean)
                else:
                    raise NotImplementedError("Only gamma dist is implemented")
        all_model_means.append(model_means)
    test_response_means = np.array(test_response_means)
    all_model_means = np.array(all_model_means)
    all_model_means = all_model_means.transpose(1, 0, 2)
    return [
        corr(test_response_means, model_means, axis=0).mean()
        for model_means in all_model_means
    ]


def average_posterior_log_prob(models, test_loader):
    device = next(models[0].parameters()).device
    all_model_log_probs = []
    for images, responses in test_loader:
        images = images.to(device)
        responses = responses.to(device)
        model_log_probs = []
        for model in models:
            model.eval()
            with torch.no_grad():
                model_dist = model(images[0])
                responses = responses + torch.finfo(torch.float32).tiny
                model_log_prob = model_dist.log_prob(responses).mean().detach().cpu()
                model_log_probs.append(model_log_prob)
        all_model_log_probs.append(model_log_probs)
    all_model_log_probs = np.array(all_model_log_probs)
    all_model_log_probs = all_model_log_probs.transpose(1, 0)
    return all_model_log_probs.mean(axis=1)


def plot_posterior_density(model, test_loader):
    device = next(model.parameters()).device
    images, responses = next(iter(test_loader))
    images = images.to(device)
    model_dist = model(images[0])
    model_mean = get_gamma_model_mean(model_dist)
    x = torch.stack(
        [
            torch.linspace(
                start=responses.min() + torch.finfo(responses.dtype).eps,
                end=responses.max(),
                steps=10_000,
            )
        ]
        * responses.shape[1]
    ).T.to(device)
    model_density = model_dist.base_dist.log_prob(x).exp().detach().cpu().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5 * 5, 5), sharey=True)
    for i, ax in enumerate(ax):
        response_plottable = responses[:, i].detach().cpu().numpy().ravel()
        density_plottable = model_density[:, i].ravel()
        x_plottable = x[:, i].detach().cpu().numpy().ravel()
        mean_plottable = model_mean[i]
        sns.histplot(
            response_plottable,
            ax=ax,
            label="Data",
            stat="probability",
            element="step",
            color="blue",
            alpha=0.3,
        )
        ax.plot(x_plottable, density_plottable, label="Model", color="red")
        ax.axvline(
            mean_plottable, label="Model mean", linestyle="dashed", color="green"
        )
        ax.set_ylim(0, 0.7)
        ax.set_xlabel("Response", fontsize=20)
        ax.set_ylabel("Probability", fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.legend(prop={"size": 10})
        sns.despine(ax=ax, trim=True)
    wandb.log({"posterior_density": wandb.Image(fig)})


def save_dict(d, filename):
    # use json
    import json

    indent = 4
    with open(filename, "w") as f:
        json.dump(d, f, indent=indent)
