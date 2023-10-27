import os
import tempfile

import datajoint as dj
import numpy as np
import torch
from gensn_experiments.monkey.posterior.build_posterior_model import (
    build_posterior_model,
)
from gensn_experiments.monkey.posterior.posterior_experiment import (
    posterior_experiment,
    posterior_experiment_compact,
)
from gensn_experiments.monkey.posterior.sample_based_posterior_experiment import (
    sample_based_nnsysident_experiment,
)
from neural_sampling_code.utilities.utilities import save_dict
from torch.utils.data import DataLoader, TensorDataset, random_split

from .load_models import (
    load_best_likelihood_model,
    load_best_prior_model,
    load_target_likelihood_model,
    load_target_prior_model,
)

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
class SampleBasedSysIdentConfig(dj.Manual):
    """
    This table contains the configuration for the sample-based posterior experiment
    """

    definition = """
    config_id: char(32)
    ---
    seed: int
    session_id: varchar(20)
    image_crop: int
    subsample: int
    scale: float
    batch_size: int
    prior_model_type: varchar(20)
    likelihood_model_type: varchar(20)
    posterior_model_config: longblob
    trainer_config: longblob
    trainer_type: varchar(30) # 'nnsysident'
"""


@schema
class SampleBasedNNSysIdentResults(dj.Computed):
    """
    This table is for training sysident models using samples
    from a generative model instead of real data, using trainer from
    nnsysident packages and the loss is predicting sampled response
    from sample stimulus.

    Note: this is only for using nnsysident trainer
    """

    force_gpu = False

    config_table = SampleBasedSysIdentConfig

    # NOTE: this should only work for a subset of the possible configs
    # Namely for:
    # 'flow' and 'exp' prior_model_type, and

    key_source_restriction = "(prior_model_type='flow' or prior_model_type='exp' or prior_model_type='half_normal' or prior_model_type = 'log_normal' or prior_model_type = 'exp_ones') and trainer_type='nnsysident'"

    key_source = (config_table & key_source_restriction).proj()

    @property
    def definition(self):
        return f"""
            -> {self.config_table.__name__}
            ---
            model: attach@external # the trained elbo model whose posterior is the sysident model
            
            tracker_output: longblob # nnsysident trainer's tracker output
            best_model_stats: longblob # nnsysident trainer's best model stats
            
            train_elbo: float
            val_elbo: float
            test_elbo: float
            train_neural_elbo: float
            val_neural_elbo: float
            test_neural_elbo: float
            train_post_ll: float
            val_post_ll: float
            test_post_ll: float
            train_post_pmf: float
            val_post_pmf: float
            test_post_pmf: float
            train_joint_ll: float
            val_joint_ll: float
            test_joint_ll: float
            
            train_correlation: float    # output from nnsysident
            validation_correlation: float   # output from nnsysident
            test_correlation: float    # output from nnsysident 

            sample_train_elbo: float
            sample_val_elbo: float
            sample_test_elbo: float
            sample_train_neural_elbo: float
            sample_val_neural_elbo: float
            sample_test_neural_elbo: float
            sample_train_post_ll: float
            sample_val_post_ll: float
            sample_test_post_ll: float
            sample_train_post_pmf: float
            sample_val_post_pmf: float
            sample_test_post_pmf: float
            sample_train_joint_ll: float
            sample_val_joint_ll: float
            sample_test_joint_ll: float

            sample_train_correlation: float   # output from nnsysident
            sample_validation_correlation: float       # output from nnsysident
            sample_test_correlation: float  # output from nnsysident

            likelihood_table_name: varchar(50)
            likelihood_config_id: char(32)
            prior_table_name: varchar(50)
            prior_config_id: char(32)   # this is the primary key for the prior table, atm experiment_id
            """

    def make(self, key):
        print("Fetching config")
        args = (self.config_table & key).fetch1()

        print("Fetching generative model")
        prior_model, likelihood_model = (
            GenerativeModelInstanceWithSessionId & (self.config_table & key)
        ).get_models()
        # add to args
        args["prior_model"] = prior_model
        args["likelihood_model"] = likelihood_model

        # get likelihood and prior config ids and table names
        gen_model_instance = (
            GenerativeModelInstanceWithSessionId & (self.config_table & key)
        ).fetch1()
        # insert those to key
        key["likelihood_table_name"] = gen_model_instance["likelihood_table_name"]
        key["likelihood_config_id"] = gen_model_instance["likelihood_config_id"]
        key["prior_config_id"] = gen_model_instance["prior_config_id"]
        key["prior_table_name"] = gen_model_instance["prior_table_name"]

        print("Fetching sample data")
        batch_size = args["batch_size"]
        train_split = 0.6
        test_split = 0.2
        seed = args["seed"]
        data_dict = (
            GenerativeModelSamplesWithSessionId & (self.config_table & key)
        ).get_dataloaders(
            batch_size=batch_size,
            train_split=train_split,
            test_split=test_split,
            seed=seed,
        )

        train_loader = data_dict["train_loader"]
        val_loader = data_dict["val_loader"]
        test_loader = data_dict["test_loader"]
        image_shape = data_dict["image_shape"]
        image_dim = data_dict["image_dim"]
        n_neurons = data_dict["n_neurons"]

        # add to args
        args["sample_train_loader"] = train_loader
        args["sample_val_loader"] = val_loader
        args["sample_test_loader"] = test_loader

        print("Building posterior model")
        # build posterior model
        model_basepath = "/src/project/computed/sysident_models/"
        posterior_model = build_posterior_model(
            posterior_model_config=args["posterior_model_config"],
            image_shape=image_shape,
            image_dim=image_dim,
            n_neurons=n_neurons,
            seed=args["seed"],
            model_basepath=model_basepath,
            session_id=args["session_id"],
        )
        args["posterior_model"] = posterior_model

        # set loss_function
        if args["posterior_model_config"]["posterior_distribution"] == "gamma":
            args["loss_function"] = "GammaLoss"
        else:
            raise NotImplementedError(
                "Only gamma posterior distribution is implemented"
            )

        # remove config_id from args because posterior_experiment doesn't take it
        args["device"] = (
            "cuda" if self.force_gpu or torch.cuda.is_available() else "cpu"
        )

        # remove keys not needed by posterior_experiment
        args.pop("config_id")
        args.pop("posterior_model_config")
        args.pop("prior_model_type")
        args.pop("likelihood_model_type")
        args.pop("trainer_type")

        # call posterior_experiment with args
        print(f"Device set: {args['device']}")
        print("Calling posterior experiment")
        model, metrics = sample_based_nnsysident_experiment(**args)

        # get device
        # save the model
        filepath = f"/tmp/{key['config_id']}.pt"
        print("Saving model")
        torch.save(model, filepath)
        # insert the results
        key["model"] = filepath
        key.update(metrics)
        print("Inserting to result")
        # insert the results into the table
        self.insert1(key, skip_duplicates=True)
        # remove the model from the filesystem
        os.remove(filepath)
        print("Experiment complete")


@schema
class SampleBasedNNSysIdentNeuripsResults(dj.Manual):
    """
    This table consists of the results from the best models from the sample-based posterior (nnsysident) experiments
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
class SampleBasedNNSysIdentNeuripsResultsRiemann(dj.Manual):
    """
    This table consists of the results from the best models from the sample-based posterior (nnsysident) experiments
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