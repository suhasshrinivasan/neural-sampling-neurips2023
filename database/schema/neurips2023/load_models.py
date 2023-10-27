import torch


def load_best_prior_model(model_type, session_id):
    """
    Load the prior model with the best validation log likelihood from the database.
    """
    from gensn_experiments.monkey.prior.schema import DequantPriorExperimentResultLog

    try:
        result = (
            DequantPriorExperimentResultLog
            & f"session_id = {session_id} and prior_type = '{model_type}'"
        ).fetch(
            order_by="val_ll DESC",
            limit=1,
            as_dict=True,
            download_path="/tmp/",
        )[
            0
        ]
    except IndexError:
        raise ValueError(
            f"No prior model of type {model_type} found for session {session_id}, aborting."
        )

    model = torch.load(result["model"])

    config_id = result["experiment_id"]

    return model, "DequantPriorExperimentResultLog", config_id


def load_target_prior_model(table_name, config_id):
    from gensn_experiments.monkey.prior import schema

    target_table = getattr(schema, table_name)
    pk = target_table.primary_key

    # we can only work with single PK attribute table at the moment
    if len(pk) > 1:
        raise ValueError(
            f"Target table {table_name} has primary key that is composite and this is not supported at the moment"
        )

    key = {pk[0]: config_id}

    model_path = (target_table & key).fetch1(
        "model",
        download_path="/tmp/",
    )

    return torch.load(model_path)


def load_best_likelihood_model(model_type, session_id):
    """
    Load the likelihood model with the best validation log likelihood from the database.
    """
    from gensn_experiments.monkey.likelihood.schema import (
        LikelihoodExperimentConfig,
        LikelihoodExperimentResult,
    )

    # in the case of Likelihood we are required to join the config table
    # and the result table in order to access the configs and the models
    likelihood_table = (LikelihoodExperimentConfig * LikelihoodExperimentResult) & (
        f"session_id = {session_id}"
    )
    # there can be several models with the same likelihood
    # for instance due to different seeds
    # hence we load the model with the highest validation log likelihood
    # don't forget limit=1, otherwise the operation would be very slow
    # TODO: at the moment we only support linear and mlp likelihoods,
    # extend this to include tconv
    if model_type == "linear":
        restriction = "nonlinearity = 'none'"
    elif model_type == "mlp":
        restriction = "nonlinearity != 'none'"
    else:
        raise ValueError(f"Likelihood type {model_type} not supported.")

    row = (likelihood_table & restriction).fetch(
        order_by="val_ll DESC", limit=1, as_dict=True, download_path="/tmp/"
    )[0]

    model = torch.load(row["model"])

    config_id = row["config_id"]

    return model, "LikelihoodExperimentResult", config_id


def load_target_likelihood_model(table_name, config_id):
    from gensn_experiments.monkey.likelihood import schema

    target_table = getattr(schema, table_name)
    pk = target_table.primary_key

    # we can only work with single PK attribute table at the moment
    if len(pk) > 1:
        raise ValueError(
            f"Target table {table_name} has primary key that is composite and this is not supported at the moment"
        )

    key = {pk[0]: config_id}

    model_path = (target_table & key).fetch1(
        "model",
        download_path="/tmp/",
    )

    return torch.load(model_path)
