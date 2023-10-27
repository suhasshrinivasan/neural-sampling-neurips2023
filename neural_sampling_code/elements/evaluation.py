import numpy as np
import torch
from tqdm import tqdm

from neuralpredictors.measures import corr


def sysident_log_prob(model, dataloader, device):
    """
    Compute the log probability of stimulus-conditioned responses under the model.

    Args:
        model (torch.nn.Module): A model with a factorized_log_prob method.
        dataloader (torch.utils.data.DataLoader): A dataloader with (images, responses) tuples.
        device (str): The device to use for computation.

    Returns:
        sum_lp_across_neurons (float): The sum of the log probabilities across neurons, mean across trials, in bits.
        mean_lp_across_neurons (float): The mean of the log probabilities across neurons, mean across trials, in bits.
        sem_lp_across_neurons (float): The standard error of the log probabilities across neurons, mean across trials, in bits.
    """
    model = model.eval()
    model = model.to(device)
    total_trials = 0
    total_lp = 0
    for _, (images, responses) in enumerate(tqdm(dataloader)):
        total_trials += responses.shape[0]
        images = images.to(device)
        responses = responses.to(device)
        lp = model.factorized_log_prob(responses, cond=images).sum(dim=0)
        total_lp = total_lp + lp
    mean_lp_across_trials = total_lp / total_trials / np.log(2)
    sum_lp_across_neurons = mean_lp_across_trials.sum().item()
    mean_lp_across_neurons = mean_lp_across_trials.mean().item()
    sem_lp_across_neurons = (
        mean_lp_across_trials.std() / np.sqrt(mean_lp_across_trials.shape[0])
    ).item()
    return sum_lp_across_neurons, mean_lp_across_neurons, sem_lp_across_neurons


def riemann_sum(dist, x, n_samples, device="cpu"):
    """Compute the Riemann sum of the log-likelihood of the responses under the distribution.

    Args:
        dist (torch.distributions.Distribution): A distribution with a log_prob method.
        responses (torch.Tensor): A tensor of responses.
        n_samples (int, optional): The number of samples to use in the Riemann sum. Defaults to 1000.

    Returns:
        torch.Tensor: The Riemann sum of the log-likelihood of the responses under the distribution.
    """
    # make responses continuous
    cont_x = (
        x[..., None]
        + torch.linspace(start=torch.finfo(x.dtype).eps, end=1, steps=n_samples + 1)
    )[..., :-1].to(device)
    cont_x = cont_x.permute(-1, *np.arange(cont_x.dim())[:-1])
    return (dist.log_prob(cont_x).exp().mean(0) + torch.finfo(cont_x.dtype).eps).log()


def sysident_log_pmf_riemann(model, dataloader, n_samples, device):
    """
    Approximate the log pmf of stimulus-conditioned responses under the model using a Riemann sum.

    Args:
        model (torch.nn.Module): A model with a factorized_log_prob method.
        dataloader (torch.utils.data.DataLoader): A dataloader with (images, responses) tuples.
        n_samples (int): The number of samples to use in the Riemann sum.
        device (str): The device to use for computation.

    Returns:
        sum_lp_across_neurons (float): The sum of the log probabilities across neurons, mean across trials, in bits.
        mean_lp_across_neurons (float): The mean of the log probabilities across neurons, mean across trials, in bits.
        sem_lp_across_neurons (float): The standard error of the log probabilities across neurons, mean across trials, in bits.
    """
    model = model.eval()
    model = model.to(device)
    total_trials = 0
    total_lp = 0
    for _, (images, responses) in enumerate(tqdm(dataloader)):
        total_trials += responses.shape[0]
        images = images.to(device)
        responses = responses.to(device)
        predicted_distribution = model.trainable_distribution.distribution(
            images
        ).base_dist
        lp = riemann_sum(predicted_distribution, responses, n_samples, device).sum(
            dim=0
        )
        total_lp = total_lp + lp
    mean_lp_across_trials = total_lp / total_trials / np.log(2)
    sum_lp_across_neurons = mean_lp_across_trials.sum().item()
    mean_lp_across_neurons = mean_lp_across_trials.mean().item()
    sem_lp_across_neurons = (
        mean_lp_across_trials.std() / np.sqrt(mean_lp_across_trials.shape[0])
    ).item()
    return sum_lp_across_neurons, mean_lp_across_neurons, sem_lp_across_neurons


def uniform_iw_bound(dist, responses, n_samples, device="cpu"):
    """Compute the importance weighted bound on the log-likelihood of the responses under the distribution.
    See Learning Discrete Distributions by Dequantization by Hoogeboom et al. (2021).
    """
    udequantized_responses = (
        responses
        + torch.rand((n_samples, *responses.shape))
        + torch.finfo(responses.dtype).eps
    ).to(device)
    return (
        dist.log_prob(udequantized_responses).exp().mean(0)
        + torch.finfo(responses.dtype).eps
    ).log()


def sysident_log_pmf_uniform(model, dataloader, n_samples, device):
    """
    Approximate the log pmf of stimulus-conditioned responses under the model using iw bound and uniform dequantization.

    Args:
        model (torch.nn.Module): A model with a factorized_log_prob method.
        dataloader (torch.utils.data.DataLoader): A dataloader with (images, responses) tuples.
        n_samples (int): The number of samples to use in the Riemann sum.
        device (str): The device to use for computation.

    Returns:
        sum_lp_across_neurons (float): The sum of the log probabilities across neurons, mean across trials, in bits.
        mean_lp_across_neurons (float): The mean of the log probabilities across neurons, mean across trials, in bits.
        sem_lp_across_neurons (float): The standard error of the log probabilities across neurons, mean across trials, in bits.
    """
    model = model.eval()
    model = model.to(device)
    total_trials = 0
    total_lp = 0
    for _, (images, responses) in enumerate(tqdm(dataloader)):
        total_trials += responses.shape[0]
        images = images.to(device)
        responses = responses.to(device)
        predicted_distribution = model.trainable_distribution.distribution(
            images
        ).base_dist
        lp = uniform_iw_bound(predicted_distribution, responses, n_samples, device).sum(
            dim=0
        )
        total_lp = total_lp + lp
    mean_lp_across_trials = total_lp / total_trials / np.log(2)
    sum_lp_across_neurons = mean_lp_across_trials.sum().item()
    mean_lp_across_neurons = mean_lp_across_trials.mean().item()
    sem_lp_across_neurons = (
        mean_lp_across_trials.std() / np.sqrt(mean_lp_across_trials.shape[0])
    ).item()
    return sum_lp_across_neurons, mean_lp_across_neurons, sem_lp_across_neurons


def get_mean_from_distribution(distribution, distribution_type):
    """
    Get the mean of a distribution.

    Args:
        distribution (torch.distributions.Distribution): A distribution.
        distribution_type (str): The type of distribution, i.e. "poisson", "normal", or "gamma".
    """
    if distribution_type == "poisson":
        return distribution.rate
    elif distribution_type == "normal":
        return distribution.loc
    elif distribution_type == "gamma":
        return distribution.concentration / distribution.rate
    else:
        raise NotImplementedError


def sysident_correlation(model, dataloader, device="cpu", distribution="poisson"):
    """
    Compute the correlation between the predicted mean and the actual responses.

    Args:
        model (gensn distribution): A model with a trainable_distribution attribute.
        dataloader (torch.utils.data.DataLoader): A dataloader with (images, responses) tuples.
        device (str): The device to use for computation.
        distribution (str): The distribution to use for the model.

    Returns:
        mean_correlation (float): The mean correlation across neurons.
        sem_correlation (float): The standard error of the correlation across neurons.
    """
    model = model.eval()
    model = model.to(device)
    all_responses = []
    all_predicted_means = []
    for _, (images, responses) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        responses = responses.to(device)
        predicted_distribution = model.trainable_distribution.distribution(
            images
        ).base_dist
        predicted_mean = get_mean_from_distribution(
            predicted_distribution, distribution
        )
        all_predicted_means.append(predicted_mean.detach().cpu().numpy())
        all_responses.append(responses.detach().cpu().numpy())
    all_responses = np.concatenate(all_responses)
    all_predicted_means = np.concatenate(all_predicted_means)
    correlations = corr(all_responses, all_predicted_means, axis=0)
    mean_correlation = correlations.mean()
    sem_correlation = correlations.std() / np.sqrt(correlations.shape[0])
    return mean_correlation, sem_correlation


def marginal_likelihood_mc(likelihood, data, prior_samples, prior_lp, device):
    """
    Compute the marginal likelihood of the tensor under the generative model specified via prior and likelihood.
    Marginal likelihood is defined as the expectation of the likelihood under the prior.
    The expectation is approximated via monte carlo integration with n_samples from the prior.

    Args:
        likelihood (gensn.distribution): The likelihood distribution.
        data (torch.Tensor): The data to compute the marginal likelihood of.
        prior_samples (torch.Tensor): Samples from the prior distribution.
        prior_lp (torch.Tensor): The log probability of the prior samples under the prior distribution.
        device (str): The device to use for computation.

    Returns:
        marginal_likelihood (float, shape BX1, where B is batch_dim): The marginal likelihood of the data under the model in bits, not reduced.
    """
    likelihood = likelihood.to(device)
    data = data.to(device)
    likelihood_lp = likelihood(data, cond=prior_samples.unsqueeze(1)).T
    return torch.logsumexp(prior_lp + likelihood_lp, dim=1)


def marginal_likelihood(prior, likelihood, dataloader, n_samples, device):
    """
    Compute the marginal likelihood of the images in the dataloader under the model.
    Marginal likelihood is defined as the expectation of the likelihood under the prior.
    The expectation is approximated via monte carlo integration with n_samples from the prior.

    Args:
        prior (gensn.distribution): The prior distribution.
        likelihood (gensn.distribution): The likelihood distribution.
        dataloader (torch.utils.data.DataLoader): A dataloader with (images, responses) tuples.
        n_samples (int): The number of samples to draw from the prior.
        device (str): The device to use for computation.

    Returns:
        marginal_likelihood (float): The marginal likelihood of the images under the model in bits, averaged across trials.
    """
    prior = prior.eval()
    prior = prior.to(device)
    likelihood = likelihood.eval()
    likelihood = likelihood.to(device)
    prior_samples = prior.sample((n_samples,))
    prior_lp = prior(prior_samples)
    total_trials = 0
    total_marginal_likelihood = 0
    for _, (images, _) in enumerate(tqdm(dataloader)):
        total_trials += images.shape[0]
        images = images.to(device)
        images = images.flatten(start_dim=1)
        marginal_likelihood = marginal_likelihood_mc(
            likelihood, images, prior_samples, prior_lp, device
        )
        total_marginal_likelihood += marginal_likelihood.sum().item()
    marginal_likelihood = total_marginal_likelihood / total_trials / np.log(2)
    return marginal_likelihood
