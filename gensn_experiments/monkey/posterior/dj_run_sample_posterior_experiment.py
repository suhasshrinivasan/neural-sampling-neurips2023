import itertools as it
from collections import OrderedDict

from database.schema.neurips2023.posterior_schema import (
    PosteriorExperimentCompactConfig3,
    SampleBasedPosteriorExperimentResult,
)
from neural_sampling_code.utilities.utilities import make_hash


print("Setting configs")
# first prepare the config and ensure the order is the same
# new set of configs for posterior mle experiment
configs = OrderedDict(
    seed=[42, 100, 213, 412, 123],
    session_id=["3631807112901"],
    image_crop=[96],
    subsample=[1],
    scale=[1.0],
    batch_size=[64, 128],
    prior_model_type=["flow", "exp", "laplace", "normal"],
    likelihood_model_type=["mlp", "linear"],
    loss_type=["elbo", "neural_elbo"],
    posterior_model_config=[
        dict(
            posterior_distribution="gamma",
            amortized_function="Stacked2dPointPooled_Gamma",
            layers=3,
            input_kern=24,
            gamma_input=10,
            gamma_readout=0.5,
            hidden_dilation=2,
            hidden_kern=9,
            hidden_channels=32,
            pretrained_model_path=None,
        ),
        dict(
            posterior_distribution="gamma",
            amortized_function="Stacked2dPointPooled_Gamma",
            layers=3,
            input_kern=24,
            gamma_input=10,
            gamma_readout=0.5,
            hidden_dilation=2,
            hidden_kern=9,
            hidden_channels=32,
            linear=True,
            pretrained_model_path=None,
        ),
        dict(
            posterior_distribution="gamma",
            amortized_function="Stacked2dPointPooled_Gamma",
            layers=3,
            input_kern=24,
            gamma_input=10,
            gamma_readout=0.5,
            hidden_dilation=2,
            hidden_kern=9,
            hidden_channels=32,
            pretrained_model_path="/src/project/computed/sysident_models/deep_gamma_model.pt",
        ),
        dict(
            posterior_distribution="gamma",
            amortized_function="Stacked2dPointPooled_Gamma",
            layers=3,
            input_kern=24,
            gamma_input=10,
            gamma_readout=0.5,
            hidden_dilation=2,
            hidden_kern=9,
            hidden_channels=32,
            linear=True,
            pretrained_model_path="/src/project/computed/sysident_models/linear_gamma_model.pt",
        ),
    ],
    l2_weight_decay=[1e-2, 1e-3],
    lr=[1e-2, 1e-4, 1e-3],
    n_epochs=[200],
    apply_early_stopping=[1],
    early_stopping_criterion=["val_loss", "val_post_loss"],
    early_stopping_threshold=[5],
    early_stopping_patience=[20],
    apply_gradient_clipping=[1],
    gradient_clipping_threshold=[10, 1000, 100, 100_000],
    apply_update_skipping=[0],
    update_skipping_threshold=[1e5],
)

# NOTE: using itertools.product is much faster
config_list = []
for values in it.product(*configs.values()):
    config = {key: value for key, value in zip(configs.keys(), values)}
    config["config_id"] = make_hash(config)
    config_list.append(config)

print("Inserting configs into PosteriorExperimentCompactConfig3")
# then insert the configs into the database
PosteriorExperimentCompactConfig3.insert(config_list, skip_duplicates=True)

print("Running experiments into PosteriorExperimentCompactResult3")
# debug
# limit=1 will run a single experiment
# SampleBasedPosteriorExperimentResult.populate(limit=1)

# in order to run all the experiments in parallel with the job manager
SampleBasedPosteriorExperimentResult.force_gpu = True  # ensure all experiments use GPU
SampleBasedPosteriorExperimentResult.populate(reserve_jobs=True, order="random")
