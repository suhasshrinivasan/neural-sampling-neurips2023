import itertools as it
from collections import OrderedDict

from experiments.monkey.likelihood.schema import (
    LikelihoodExperimentConfig,
    LikelihoodExperimentResult,
)
from neural_sampling_code.utilities.utilities import make_hash

# first prepare the config and ensure the order is the same
configs = OrderedDict(
    seed=[42, 100, 213, 412, 123],
    session_id=[
        "3636034866307",
        "3635178040531",
        "3637851724731",
        "3638973674012",
        "3637248451650",
        "3638885582960",
        "3638373332053",
        "3638541006102",
        "3633364677437",
        "3640011636703",
        "3639664527524",
        "3634744023164",
        "3637161140869",
        "3638456653849",
        "3635949043110",
        "3632932714885",
        "3634658447291",
        "3637333931598",
        "3638802601378",
        "3639749909659",
        "3639060843972",
        "3639406161189",
        "3639492658943",
        "3634142311627",
        "3632669014376",
        "3634055946316",
        "3637760318484",
        "3638367026975",
    ],
    image_crop=[96],
    subsample=[1],
    scale=[1.0],
    batch_size=[128],
    n_layers=[2, 4],
    nonlinearity=["relu", "leaky_relu", "none"],
    dropout_rate=[0.0, 0.5, 0.8],
    nonneg_std_transform=["exp"],
    init_std=[1e-4, 1e-3],
    l2_weight_decay=[1e-1, 1e-3],
    lr=[1e-4, 1e-3],
    n_epochs=[200],
    early_stopping_threshold=[10],
    gradient_clipping_threshold=[1e6],
    early_stopping_patience=[10],
)

config_list = []
for values in it.product(*configs.values()):
    config = {key: value for key, value in zip(configs.keys(), values)}
    config["config_id"] = make_hash(config)
    config_list.append(config)

# then insert the configs into the database
LikelihoodExperimentConfig.insert(config_list, skip_duplicates=True)

# then run the experiments with LikelihodExperimentResult.populate(limit=1)
# limit=1 will run a single experiment
# LikelihoodExperimentResult.populate(limit=1)

# in order to run all the experiments in parallel with the job manager
LikelihoodExperimentResult.populate(reserve_jobs=True, order="random")
