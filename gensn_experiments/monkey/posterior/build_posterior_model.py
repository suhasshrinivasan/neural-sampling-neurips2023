from copy import deepcopy
from pathlib import Path

import torch
from neural_sampling_code.elements.conditioned_parameters import ConcRateMLPFlexible
from neural_sampling_code.elements.layers import View
from system_identification.monkey_sysident_experiment import build_sysident_model
from torch import nn

import gensn.distributions as G
from nnsysident.models.models import Stacked2dPointPooled_Gamma


def build_posterior_model(
    posterior_model_config,
    image_shape,
    image_dim,
    n_neurons,
    seed,
    model_basepath,
    session_id=None,
):
    """
    Builds the posterior model based on the posterior_model_config.

    Args:
        - posterior_model_config is a dictionary with the following compulsory keys:
            - posterior_distribution: str, name of the posterior distribution
            - amortized_function: str, name of the amortized function
            Based on the values of these keys, the function will build the posterior model,
            and will expect further keys based on the posterior distribution and the amortized function.
            - pretrained_model_path: str, path to the pretrained model
            If None, the model will be trained from scratch.
        - image_shape is a tuple, the shape of the image (channels, height, width)
        - image_dim is an int, the flattened dimension of the image
        - n_neurons is an int, the number of neurons
        - seed is an int, the seed for the random number generator for reproducibility
        - model_basepath is a str, the basepath for the pretrained posterior model state dict
        - session_id is a str, the session id of the dataset, default is None
    """

    if posterior_model_config["posterior_distribution"] == "gamma":
        # set up the amortized function
        if posterior_model_config["amortized_function"] == "ConcRateMLPFlexible":
            amortized_model = ConcRateMLPFlexible(
                in_features=image_dim,
                out_features=n_neurons,
                n_layers=posterior_model_config["n_layers"],
                nonlinearity=posterior_model_config["nonlinearity"],
                dropout_rate=posterior_model_config["dropout_rate"],
                init_std=posterior_model_config["init_std"],
                nonneg_transform=posterior_model_config["nonneg_transform"],
            )
            # assign model to load, see below
            model_to_load = amortized_model
        elif (
            posterior_model_config["amortized_function"] == "Stacked2dPointPooled_Gamma"
        ):
            amortized_model_config = deepcopy(posterior_model_config)
            # remove keys that are not needed for the amortized model
            _ = amortized_model_config.pop("posterior_distribution")
            _ = amortized_model_config.pop("amortized_function")
            data_info = {
                f"{session_id}": {
                    "input_dimensions": (64,) + image_shape,
                    "input_channels": image_shape[0],
                    "output_dimension": n_neurons,
                }
            }
            # update data_info in amortized_model_config
            # update because there might be other data_infos present
            amortized_model_config.setdefault("data_info", {}).update(data_info)
            # use min_rate=0.2, max_concentration=100
            min_rate = 0
            max_concentration = 100
            eps = 1e-6
            amortized_model_base = Stacked2dPointPooled_Gamma().build_model(
                dataloaders=None,
                eps=eps,
                min_rate=min_rate,
                max_concentration=max_concentration,
                seed=seed,
                **amortized_model_config,
            )
            # amortized model expects an image of shape (channels, height, width)
            # whereas the rest of the generative model at the moment expects
            # a flattened image, hence view the flattened image
            amortized_model = nn.Sequential(
                View(image_shape),
                amortized_model_base,
            )
            # assign model to load, in this case its base model without the sequential
            model_to_load = amortized_model_base
        else:
            raise ValueError(
                f"Amortized function {posterior_model_config['amortized_function']} not supported."
            )
        posterior_model = G.IndependentGamma(
            _parameters=amortized_model,
        )
    elif posterior_model_config["posterior_distribution"] == "poisson":
        if (
            posterior_model_config["amortized_function"]
            == "Stacked2dPointPooled_Poisson"
        ):
            amortized_model_config = deepcopy(posterior_model_config)
            # remove keys that are not needed for the amortized model
            _ = amortized_model_config.pop("posterior_distribution")
            _ = amortized_model_config.pop("amortized_function")
            data_info = {
                f"{session_id}": {
                    "input_dimensions": (64,) + image_shape,
                    "input_channels": image_shape[0],
                    "output_dimension": n_neurons,
                }
            }
            # update data_info in amortized_model_config
            # update because there might be other data_infos present
            amortized_model_config.setdefault("data_info", {}).update(data_info)
            amortized_model_base = build_sysident_model(
                dataloaders=None,
                seed=seed,
                distribution_type="poisson",
                model_config=amortized_model_config,
            )
            # amortized model expects an image of shape (channels, height, width)
            # whereas the rest of the generative model at the moment expects
            # a flattened image, hence view the flattened image
            amortized_model = nn.Sequential(
                View(image_shape),
                amortized_model_base,
            )
            # assign model to load, in this case its base model without the sequential
            model_to_load = amortized_model_base
        else:
            raise ValueError(
                f"Amortized function {posterior_model_config['amortized_function']} not supported."
            )
        posterior_model = G.IndependentPoisson(
            _parameters=amortized_model,
        )
    else:
        raise ValueError(
            f"Posterior distribution {posterior_model_config['posterior_distribution']} not supported."
        )
    if posterior_model_config.get("pretrained_model_path", None):
        corrected_path_string = (
            posterior_model_config["pretrained_model_path"][:-3] + f"_{session_id}.pt"
        )
        amortized_model_path = Path(corrected_path_string)
        if not amortized_model_path.is_absolute():
            amortized_model_path = Path(model_basepath) / amortized_model_path
        model_to_load.load_state_dict(
            torch.load(amortized_model_path, map_location="cpu")
        )
    return posterior_model
