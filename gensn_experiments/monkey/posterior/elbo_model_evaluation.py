import numpy as np
import torch
import wandb
from neural_sampling_code.elements import loss_functions

from neuralpredictors.measures import corr


def evaluate_all_session_batches_with_loss_per_neuron(
    model, data_loader, device, loss_fn, label=""
):
    model.to(device)
    model.eval()
    total_log_lik = 0
    n_trials = 0
    total_batches = 0
    all_sessions = list(data_loader.keys())
    with torch.no_grad():
        for session in all_sessions:
            for batch in data_loader[session]:
                total_batches += 1
                batch = [element.to(device) for element in batch]
                n_neurons = batch[1].shape[-1]
                n_trials = batch[1].shape[0]
                log_lik = -loss_fn(model, batch, reduction="sum") / n_trials / n_neurons
                total_log_lik += log_lik.item()
    return total_log_lik


def evaluate_all_batches_with_loss(model, data_loader, device, loss_fn, label=""):
    """
    Evaluate the model on all batches in the data loader.

    Args:
        model (gensn.variational.ELBOMarginal): model to evaluate
        data_loader (torch.utils.data.DataLoader): data loader
        device (str): device
        loss_fn (neural_sampling_code.elements.loss_functions): loss function

    Returns:
        float: loss averaged over all batches in the data loader
    """
    # print(f"Computing {loss_fn}")
    model.to(device)
    model.eval()
    total_log_lik = 0
    n_trials = 0
    total_batches = 0
    with torch.no_grad():
        for batch in data_loader:
            total_batches += 1
            batch = [element.to(device) for element in batch]
            log_lik = -loss_fn(model, batch, reduction="sum")
            n_trials += batch[0].shape[0]
            total_log_lik += log_lik.item()
    return total_log_lik / n_trials


def get_gamma_correlation(model, data_loader, device):
    """
    Computes the correlation between a gamma model and the responses in the data loader.
    """
    all_responses = []
    all_gamma_means = []
    for images, responses in data_loader:
        images = images.to(device)
        responses = responses.to(device)
        gamma_distribution = model.trainable_distribution.distribution(images)
        gamma_means = (
            gamma_distribution.base_dist.concentration
            / gamma_distribution.base_dist.rate
        )
        gamma_means = gamma_means.detach().cpu().numpy()
        responses = responses.detach().cpu().numpy()
        all_responses.append(responses)
        all_gamma_means.append(gamma_means)
    all_responses = np.concatenate(all_responses)
    all_gamma_means = np.concatenate(all_gamma_means)
    return corr(all_responses, all_gamma_means, axis=0).mean()


def get_gamma_mean_match(model, data_loader, device):
    """
    Computes the mean match between a gamma model and the responses in the data loader.
    """
    all_responses = []
    all_gamma_means = []
    for images, responses in data_loader:
        images = images.to(device)
        responses = responses.to(device)
        gamma_distribution = model.trainable_distribution.distribution(images)
        gamma_means = (
            gamma_distribution.base_dist.concentration
            / gamma_distribution.base_dist.rate
        )
        gamma_means = gamma_means.detach().cpu().numpy()
        responses = responses.detach().cpu().numpy()
        all_responses.append(responses)
        all_gamma_means.append(gamma_means)
    all_responses = np.concatenate(all_responses)
    all_gamma_means = np.concatenate(all_gamma_means)
    return -np.sqrt(np.mean((all_responses - all_gamma_means) ** 2))


def compute_elbo_metrics(
    elbo_model,
    dataloader,
    device,
    posterior_distribution,
    n_samples=100,
):
    """
    Compute the ELBO, neural ELBO, posterior log likelihood, posterior pmf, and
    joint log likelihood of the model on the data in the dataloader.

    Args:
        elbo_model (gensn.variational.ELBOMarginal): model to evaluate
        dataloader (torch.utils.data.DataLoader): data loader
        device (str): device
        posterior_distribution (str): posterior distribution
        n_samples (int, optional): number of samples to use for the ELBO. Defaults to 100.
    Returns:
        dict: dictionary with the ELBO, neural ELBO, posterior log likelihood, posterior pmf, and
        joint log likelihood of the model on the data in the dataloader.
    Remarks:
        TODO: works only for gamma posterior at the moment
    """
    elbo_model.n_samples = n_samples
    elbo = evaluate_all_batches_with_loss(
        elbo_model, dataloader, device, loss_fn=loss_functions.elbo
    )

    neural_elbo = evaluate_all_batches_with_loss(
        elbo_model,
        dataloader,
        device,
        loss_fn=loss_functions.neural_elbo_with_uniform_noise,
    )

    # HACK TO AVOID NANs, posinfs, and neginfs
    elbo = torch.nan_to_num(torch.tensor(elbo)).item()
    neural_elbo = torch.nan_to_num(torch.tensor(neural_elbo)).item()

    if posterior_distribution == "gamma":
        post_ll = evaluate_all_batches_with_loss(
            elbo_model,
            dataloader,
            device,
            loss_fn=loss_functions.posterior_mle_with_uniform_noise_on_elbo,
        )
        post_pmf = evaluate_all_batches_with_loss(
            elbo_model,
            dataloader,
            device,
            loss_fn=loss_functions.gamma_post_log_pmf,
        )
        correlation = get_gamma_correlation(elbo_model.posterior, dataloader, device)
    else:
        raise NotImplementedError

    joint_ll = evaluate_all_batches_with_loss(
        elbo_model.joint, dataloader, device, loss_fn=loss_functions.joint_mle
    )
    return dict(
        elbo=elbo,
        neural_elbo=neural_elbo,
        post_ll=post_ll,
        post_pmf=post_pmf,
        correlation=correlation,
        joint_ll=joint_ll,
    )


def compute_elbo_metrics_on_dataloaders(
    elbo_model, device, posterior_distribution, n_samples=100, **dataloaders_kwargs
):
    """
    Compute the ELBO, neural ELBO, posterior log likelihood, posterior pmf, and
    joint log likelihood of the model on each dataloader in dataloaders_kwargs.

    Args:
        elbo_model (gensn.variational.ELBOMarginal): model to evaluate
        device (str): device
        posterior_distribution (str): posterior distribution
        n_samples (int, optional): number of samples to use for the ELBO. Defaults to 100.
        **dataloaders_kwargs: data loaders with names as keys.
            The names will be used as prefixes for the metrics.
            Example dataloader_kwargs:
            dataloaders_kwargs={
                "sample_train_loader": sample_train_loader,
                "sample_val_loader": sample_val_loader,
                "sample_test_loader": sample_test_loader,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
            }
            compute_elbo_metrics_on_dataloaders(..., **dataloaders_kwargs)
    """
    all_metrics = {}
    for key in dataloaders_kwargs:
        dataloader = dataloaders_kwargs[key]
        metrics = compute_elbo_metrics(
            elbo_model,
            dataloader,
            device,
            posterior_distribution,
            n_samples=n_samples,
        )
        dataloader_name = key
        # add dataloader name as prefix to the metrics
        metrics = {f"{dataloader_name}_{k}": v for k, v in metrics.items()}
        all_metrics.update(metrics)
    return all_metrics


def get_elbo_parts(
    elbo_model, data_loader, device, n_samples=1, entropy_analytical=False, label=""
):
    """
    Returns the components of elbo individually
    i.e., elbo = <log_recon + log_prior - log_post>_q
    where q is the approximate posterior.
    -<log_post>_q is the entropy of the approximate posterior
    which is analyically computable for some distributions.

    Args:
        elbo_model: the model that computes the elbo
        data_loader: the data_loader to use
        device (torch.device): the device to use
        n_samples (int, optional): number of samples to draw from the posterior
            to compute elbo. Defaults to 1_000.
        entropy_analytical (bool, optional): whether to use analytical entropy

    Returns:
        tuple: (log_recon, log_prior, -log_post, post_entropy)
    """
    elbo_model.eval()
    elbo_model.to(device)
    total_log_recon = 0
    total_log_prior = 0
    total_log_post = 0
    total_post_entropy = 0
    n_trials = 0
    if torch.no_grad():
        for images, responses in data_loader:
            n_trials += images.shape[0]
            images = images.to(device)
            responses = responses.to(device)
            wandb.log(
                {
                    f"{label}/images": wandb.Histogram(
                        images.detach().cpu().numpy().flatten()
                    )
                }
            )
            wandb.log(
                {
                    f"{label}/responses": wandb.Histogram(
                        responses.detach().cpu().numpy().flatten()
                    )
                }
            )
            # flatten the images in the batch
            images = images.flatten(start_dim=1)
            # draw samples from the posterior
            post_dist = elbo_model.posterior.trainable_distribution.distribution(
                cond=images
            )
            z_samples = post_dist.rsample((n_samples,))
            # z_samples = z_samples.clamp(max=30)

            wandb.log(
                {
                    f"{label}/z_samples": wandb.Histogram(
                        z_samples.detach().cpu().numpy().flatten()
                    )
                }
            )
            # compute the log prob of the samples under the posterior
            log_post = post_dist.log_prob(z_samples).mean(dim=0).sum()
            total_log_post += log_post
            # log_post = post_dist.log_prob(z_samples).mean(dim=0)
            if entropy_analytical:
                post_entropy = post_dist.entropy().sum()
                total_post_entropy += post_entropy
                # post_entropy = post_dist.entropy().mean(dim=0)
                # total_post_entropy += post_entropy.sum()
            # compute the log prob of the samples under the prior
            log_prior = elbo_model.joint.prior(z_samples).mean(dim=0).sum()
            total_log_prior += log_prior
            # log_prior = elbo_model.joint.prior(z_samples).mean(dim=0)
            # compute the log prob of the samples under the likelihood
            log_conditional = (
                elbo_model.joint.conditional(images, cond=z_samples).mean(dim=0).sum()
            )

            cond_samples = (
                elbo_model.joint.conditional.trainable_distribution.distribution(
                    cond=z_samples
                ).base_dist
            )
            cond_samples_mean = cond_samples.mean
            cond_samples_std = cond_samples.scale
            cond_data = (
                elbo_model.joint.conditional.trainable_distribution.distribution(
                    cond=responses
                ).base_dist
            )
            cond_data_mean = cond_data.mean
            cond_data_std = cond_data.scale

            wandb.log(
                {
                    f"{label}/cond_samples_mean": wandb.Histogram(
                        cond_samples_mean.detach().cpu().numpy().flatten()
                    ),
                    f"{label}/cond_samples_std": wandb.Histogram(
                        cond_samples_std.detach().cpu().numpy().flatten()
                    ),
                    f"{label}/cond_data_mean": wandb.Histogram(
                        cond_data_mean.detach().cpu().numpy().flatten()
                    ),
                    f"{label}/cond_data_std": wandb.Histogram(
                        cond_data_std.detach().cpu().numpy().flatten()
                    ),
                }
            )

            total_log_recon += log_conditional

            wandb.log(
                {
                    f"{label}/log_post": log_post.item()
                    / responses.shape[0]
                    / responses.shape[1],
                    f"{label}/log_prior": log_prior.item()
                    / responses.shape[0]
                    / responses.shape[1],
                    f"{label}/log_conditional": log_conditional.item()
                    / images.shape[0]
                    / images.shape[1],
                    f"{label}/elbo": (
                        log_conditional.item() + log_prior.item() - log_post.item()
                    )
                    / images.shape[0]
                    / images.shape[1],
                }
            )

            wandb.log(
                {
                    f"{label}/mean_diff": (z_samples - responses).mean().abs().item(),
                    f"{label}/var_diff": (z_samples - responses).var().abs().item(),
                }
            )

            # log_recon = elbo_model.joint.conditional(images, cond=z_samples).mean(dim=0)
    # compute the average
    avg_log_recon = (total_log_recon / n_trials).item()
    avg_log_prior = (total_log_prior / n_trials).item()
    avg_log_post = (total_log_post / n_trials).item()
    if entropy_analytical:
        avg_post_entropy = (total_post_entropy / n_trials).item()
    else:
        avg_post_entropy = None
    return (
        avg_log_recon,
        avg_log_prior,
        avg_log_post,
        avg_post_entropy,
    )
