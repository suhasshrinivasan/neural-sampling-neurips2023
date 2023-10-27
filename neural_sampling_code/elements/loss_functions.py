"""
Typical loss functions for MLE training and evaluation.
"""
import torch


def model_mle(model, batch, obs_ids=None, cond_ids=None, reduction="mean"):
    """
    Function that expects model.forward to return the log prob of its inputs
    under the model.

    Args:
        model: model to evaluate, that has a forward method that returns the log
            prob of its inputs under the model
        batch: batch of data to evaluate the model on
        obs_ids: indices of the batch that correspond to the observed variables
        cond_ids: indices of the batch that correspond to the conditioning variables
        reduction: how to reduce the loss over the batch. Can be "mean", "sum", or "none"

    Returns:
        loss: the loss value, which is the negative log prob of the batch under the model

    Remarks:
        This only works for models that return the log prob of the inputs under the model.
    """
    obs = batch if obs_ids is None else tuple(batch[i] for i in obs_ids)
    cond = None if cond_ids is None else tuple(batch[i] for i in cond_ids)
    # negation is because we want to minimize the loss
    if reduction == "mean":
        return -model(*obs, cond=cond).mean()
    elif reduction == "sum":
        return -model(*obs, cond=cond).sum()
    elif reduction == "none":
        return -model(*obs, cond=cond)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


# Below functions are specific to the monkey and mouse datasets
# and also to the specific to the models used for neural sampling code project
# In general, the monkey and mouse datasets are structured as follows:
# batch[0] is the image in the form of a tensor of shape (N, C=1, H, W)
# batch[1] is the neuron in the form of a tensor of shape (N, N_neurons)
# Our models require the image to be flattened to a vector of shape (N, C*H*W)


# TODO:
# Below functions are specific to the monkey and mouse datasets
# and also to the specific to the models used for neural sampling code project
# The names are a bit misleading since the functions are named generically
# such as prior_mle and likelihood_mle.


def image_flattened_batch(batch):
    """
    Flattens the image in the batch to a vector

    Remarks:
        This is specific to the monkey and mouse datasets
        where batch[0] is the image in the form of a tensor of shape (N, C, H, W)
        and batch[1] is the neuron in the form of a tensor of shape (N, N_neurons).

        Neural sampling models require the image to be flattened to a vector
        of shape (N, C*H*W).
    """
    return [batch[0].flatten(start_dim=1), batch[1]]


def uniform_dequantize_spikes(spikes):
    """
    Dequantizes the spike counts by adding uniform noise to the spike counts.
    In addition to the uniform noise, a small value (eps) is added to avoid
    zero responses such that continuous densities like Gamma can be used to evaluate
    its log prob.
    """
    # `uniform_dequantize_spikes` is a function that dequantizes the spike counts by adding uniform
    # noise to the spike counts. In addition to the uniform noise, a small value (eps) is added to
    # avoid zero responses such that continuous densities like Gamma can be used to evaluate its log
    # prob. The line `return spikes + torch.rand_like(spikes) + torch.finfo(spikes.dtype).eps` adds
    # uniform noise and eps to the spike counts. `torch.rand_like(spikes)` generates a tensor of the
    # same shape as `spikes` with values sampled from a uniform distribution between 0 and 1.
    # `torch.finfo(spikes.dtype).eps` returns the smallest representable number for the data type of
    # `spikes`, which is added to avoid zero responses.
    return spikes + torch.rand_like(spikes) + torch.finfo(spikes.dtype).eps


def prior_mle(model, batch, reduction="mean"):
    """
    Calls model_mle by passing obs_ids=[1] where batch[1] is neuronal response.
    The function is so named since the prior is a model over latent variables
    interpreted as neuronal responses.
    """
    return model_mle(model, batch, obs_ids=[1], cond_ids=None, reduction=reduction)


def likelihood_mle(model, batch, reduction="mean", flatten_image=True):
    """
    Calls model_mle by passing obs_ids=[0] where batch[0] is the image and
    cond_ids=[1] where batch[1] is the neuronal response.
    The function is so named since the likelihood is a model over the image
    given the neuronal responses.
    """
    if flatten_image:
        batch = image_flattened_batch(batch)
    return model_mle(
        model,
        batch,
        obs_ids=[0],
        cond_ids=[1],
        reduction=reduction,
    )


def elbo(model, batch, reduction="mean", flatten_image=True):
    """
    Calls model_mle by passing obs_ids=[0] where batch[0] is the image.
    The function is so named since the elbo is a model over the image.
    """
    if flatten_image:
        batch = image_flattened_batch(batch)
    return model_mle(
        model,
        batch,
        obs_ids=[0],
        cond_ids=None,
        reduction=reduction,
    )


def joint_mle(model, batch, reduction="mean", flatten_image=True):
    """
    Calls model_mle by passing obs_ids=[1, 0] where batch[0] is the image and
    batch[1] is the neuronal response.

    Remarks:
        This model expects a joint distribution model where the first element
        of the batch is expected to be a sample of the prior and the second
        element of the batch is expected to be a sample of the likelihood.
    """
    if flatten_image:
        batch = image_flattened_batch(batch)
    return model_mle(
        model,
        batch,
        obs_ids=[1, 0],
        cond_ids=None,
        reduction=reduction,
    )


def posterior_mle(model, batch, reduction="mean", flatten_image=True):
    """
    Calls model_mle by passing obs_ids=[1] where batch[1] is the neuronal response
    and cond_ids=[0] where batch[0] is the image.
    The function is so named since the posterior is a model over the latent
    variables (neuronal responses) given the image.

    Remarks:
        Note that this model does not take care of the possibility that the posterior
        model is continuous and neuronal responses are discrete.

        If this case needs to be handled then use posterior_mle_with_uniform_noise below.
    """
    if flatten_image:
        batch = image_flattened_batch(batch)
    return model_mle(
        model,
        batch,
        obs_ids=[1],
        cond_ids=[0],
        reduction=reduction,
    )


def posterior_mle_with_uniform_noise(
    model, batch, reduction="mean", flatten_image=True
):
    """
    Calls model_mle by passing obs_ids=[1] where batch[1] is the neuronal response
    and cond_ids=[0] where batch[0] is the image. we add uniform noise to them to make them continuous, to be
    evaluated by a continuous posterior model correctly.
    """
    if flatten_image:
        batch = image_flattened_batch(batch)
    dequantized_spikes = uniform_dequantize_spikes(batch[1])
    batch = [batch[0], dequantized_spikes]
    return model_mle(model, batch, obs_ids=[1], cond_ids=[0], reduction=reduction)


def posterior_mle_with_constant_noise(
    model, batch, reduction="mean", flatten_image=True
):
    """
    Calls model_mle by passing obs_ids=[1] where batch[1] is the neuronal response
    and cond_ids=[0] where batch[0] is the image. we add uniform noise to them to make them continuous, to be
    evaluated by a continuous posterior model correctly.
    """
    if flatten_image:
        batch = image_flattened_batch(batch)
    dequantized_spikes = 1e-10
    batch = [batch[0], dequantized_spikes]
    return model_mle(model, batch, obs_ids=[1], cond_ids=[0], reduction=reduction)


def posterior_mle_with_uniform_noise_on_elbo(
    model, batch, reduction="mean", flatten_image=True
):
    """
    Calls model_mle by passing obs_ids=[1] where batch[1] is the neuronal response
    and cond_ids=[0] where batch[0] is the image and by passing the model's
    posterior model as the model for model_mle.

    The model argument of this function is expected to be an
    """
    if flatten_image:
        batch = image_flattened_batch(batch)
    dequantized_spikes = uniform_dequantize_spikes(batch[1])
    batch = [batch[0], dequantized_spikes]
    return model_mle(
        model.posterior, batch, obs_ids=[1], cond_ids=[0], reduction=reduction
    )


def neural_elbo(model, batch, reduction="mean", flatten_image=True):
    """
    Neural ELBO is the sum of the ELBO and the posterior MLE.

    Remarks:
        This function calls posterior_mle without adding uniform noise to the
        neuronal responses.
    """
    return elbo(
        model, batch, reduction=reduction, flatten_image=flatten_image
    ) + posterior_mle(
        model.posterior, batch, reduction=reduction, flatten_image=flatten_image
    )


def neural_elbo_with_uniform_noise(model, batch, reduction="mean", flatten_image=True):
    """
    Neural ELBO is the sum of the ELBO and the posterior MLE. This function
    adds uniform noise the neuronal responses before computing the posterior MLE.
    """
    return elbo(
        model, batch, reduction=reduction, flatten_image=flatten_image
    ) + posterior_mle_with_uniform_noise(
        model.posterior, batch, reduction=reduction, flatten_image=flatten_image
    )


def gamma_post_log_pmf(elbo_model, batch, reduction="mean", n_samples=1000):
    """
    Computes the log pmf loss of the gamma distributed posterior of the elbo model via numerical integration.
    Args:
        elbo_model: a gensn.variational.ELBOMarginal object
        batch: a batch of [images, responses] of shape (N, H, W), (N, N_neurons)
        reduction: reduction method across the batch
        n_samples: number of samples to use for numerical integration to compute the pmf
    Returns:
        The pmf of the gamma distributed posterior of the elbo model as evaluated
        on the neuronal responses in the batch.
    """
    batch = image_flattened_batch(batch)
    images = batch[0]
    responses = batch[1]
    device = images.device
    # the posterior distribution is expected to be a gamma distribution
    # with concentration and rate parameters
    # we compute the pmf of the gamma distribution via numerical integration (Riemann sum)
    # of the pdf of the gamma distribution
    # expand responses to (n_samples, N, N_neurons) where n_samples is the number of samples
    # to use for numerical integration, and the samples are linspace between 0 and 1
    responses = responses.unsqueeze(-1) + torch.linspace(
        torch.finfo(responses.dtype).tiny, 1, n_samples
    ).to(device)
    # above produces a tensor of shape (N, N_neurons, n_samples)
    # we need to permute it to (n_samples, N, N_neurons)
    responses = responses.permute(2, 0, 1)
    # compute the pdf of the gamma distribution
    log_pdf = elbo_model.posterior(responses, cond=images)
    # compute the pmf of the gamma distribution via numerical integration
    # of the pdf of the gamma distribution
    log_pmf = log_pdf.mean(dim=0)
    # apply the reduction and return -log_pmf as loss
    if reduction == "mean":
        return -log_pmf.mean()
    elif reduction == "sum":
        return -log_pmf.sum()
    elif reduction == "none":
        return -log_pmf


def gamma_post_log_pmf_posterior(
    posterior_model, batch, reduction="mean", n_samples=1000
):
    """
    Computes the log pmf loss of the gamma distributed posterior of the elbo model via numerical integration.
    Args:
        posterior_model: a gensn.variational.ELBOMarginal object
        batch: a batch of [images, responses] of shape (N, H, W), (N, N_neurons)
        reduction: reduction method across the batch
        n_samples: number of samples to use for numerical integration to compute the pmf
    Returns:
        The pmf of the gamma distributed posterior of the elbo model as evaluated
        on the neuronal responses in the batch.
    """
    batch = image_flattened_batch(batch)
    images = batch[0]
    responses = batch[1]
    device = images.device
    # the posterior distribution is expected to be a gamma distribution
    # with concentration and rate parameters
    # we compute the pmf of the gamma distribution via numerical integration (Riemann sum)
    # of the pdf of the gamma distribution
    # expand responses to (n_samples, N, N_neurons) where n_samples is the number of samples
    # to use for numerical integration, and the samples are linspace between 0 and 1
    responses = responses.unsqueeze(-1) + torch.linspace(
        torch.finfo(responses.dtype).tiny, 1, n_samples
    ).to(device)
    # above produces a tensor of shape (N, N_neurons, n_samples)
    # we need to permute it to (n_samples, N, N_neurons)
    responses = responses.permute(2, 0, 1)
    # compute the pdf of the gamma distribution
    log_pdf = posterior_model(responses, cond=images)
    # compute the pmf of the gamma distribution via numerical integration
    # of the pdf of the gamma distribution
    log_pmf = log_pdf.mean(dim=0)
    # apply the reduction and return -log_pmf as loss
    if reduction == "mean":
        return -log_pmf.mean()
    elif reduction == "sum":
        return -log_pmf.sum()
    elif reduction == "none":
        return -log_pmf


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
