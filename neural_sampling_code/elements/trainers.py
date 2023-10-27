from abc import ABC, abstractmethod

import numpy as np
import torch

from ..ml_tools.training_helpers import EarlyStopper
from .loss_functions import model_mle
from gensn.variational import ELBOMarginal
from neuralpredictors.layers.encoders.distribution_encoders import GammaEncoder


class Trainer:
    def __init__(
        self,
        optimizer,
        lr,
        logger,
        device,
        compute_loss=None,
        apply_early_stopping=False,
        compute_early_stopping_loss=None,
        early_stopping_threshold=None,
        early_stopping_patience=None,
        apply_gradient_clipping=False,
        gradient_clipping_threshold=None,
        apply_update_skipping=False,
        update_skipping_threshold=None,
        detach_core=False,
    ):
        assert not detach_core, "detach_core is not implemented yet"
        # TODO:
        # detach_core is specific to system identification
        # detach_core needs to be passed to the model along with images
        # change loss_functions to accept detach_core as an argument
        # also make the model accept data_key which specifies which session the model
        # readout is being trained on
        self.optimizer = optimizer
        self.lr = lr
        self.logger = logger
        self.device = device
        if compute_loss is None:
            compute_loss = model_mle
        self.compute_loss = compute_loss
        self.apply_early_stopping = apply_early_stopping
        # by default the early stopping loss is the same as the validation loss
        # but it can be different and must be specified via the compute_early_stopping_loss argument
        # if early stopping is enabled and compute_early_stopping_loss is None (default),
        # then the early stopping loss is the same as the validation loss
        if compute_early_stopping_loss is None:
            compute_early_stopping_loss = self.compute_loss
        self.compute_early_stopping_loss = compute_early_stopping_loss
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience = early_stopping_patience
        self.early_stopper = EarlyStopper(
            patience=early_stopping_patience, min_delta=early_stopping_threshold
        )
        self.apply_gradient_clipping = apply_gradient_clipping
        self.gradient_clipping_threshold = gradient_clipping_threshold
        self.apply_update_skipping = apply_update_skipping
        self.update_skipping_threshold = update_skipping_threshold
        self.detach_core = detach_core

    def __call__(self, model, train_loader, val_loader, n_epochs):
        return self.train_and_validate(model, train_loader, val_loader, n_epochs)

    def __repr__(self):
        return f"""Trainer(
                optimizer={self.optimizer},
                lr={self.lr},
                logger={self.logger},
                device={self.device},
                compute_loss={self.compute_loss.__name__},
                apply_early_stopping={self.apply_early_stopping},
                compute_early_stopping_loss={self.compute_early_stopping_loss.__name__},
                early_stopping_threshold={self.early_stopping_threshold},
                early_stopping_patience={self.early_stopping_patience},
                apply_gradient_clipping={self.apply_gradient_clipping},
                gradient_clipping_threshold={self.gradient_clipping_threshold},
                apply_update_skipping={self.apply_update_skipping},
                update_skipping_threshold={self.update_skipping_threshold},
                )"""

    def train_and_validate(self, model, train_loader, val_loader, n_epochs):
        model.to(self.device)
        # This is the first time the trainer sees the model
        # If using a system identification model, then the model itself includes a regularizer
        # In that case, we need to update our optimizer to not include the regularizer in the parameters
        # that are being optimized
        # TODO: this is model specific
        # if has_regularizer(model):
        #     self.optimizer.param_groups[0]["weight_decay"] = 0
        train_losses = []
        train_losses_batchwise = []
        val_losses = []
        val_losses_batchwise = []
        early_stopping_losses = []
        early_stopping_losses_batchwise = []
        for epoch in range(n_epochs):
            # --- Train and validate the model for one epoch ---

            # train_loss is the average loss over the training set (scalar)
            # train_loss_batchwise is the loss for each batch in the training set (list of scalars)
            train_loss, train_loss_batchwise = self.train_epoch(
                model, train_loader, epoch
            )
            train_losses.append(train_loss)
            train_losses_batchwise.append(train_loss_batchwise)
            # note that self.validate_epoch also takes in the loss_computation_fn as an argument
            # and here, we need the validation loss to be computed the same way as the training loss
            # hence we use self.compute_loss
            val_loss, val_loss_batchwise = self.validate_epoch(
                model, val_loader, epoch, self.compute_loss
            )
            val_losses.append(val_loss)
            val_losses_batchwise.append(val_loss_batchwise)

            # --- Add the train and validation losses to a dict to log for this epoch ---
            count_phrase = f"Epoch {epoch + 1}/{n_epochs}"
            print(f"{count_phrase}")
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            # --- If early stopping is enabled, then compute the early stopping loss ---
            # and check if we should stop

            if self.apply_early_stopping:
                # if the early stopping loss computation is the same as the model loss computation
                # then we just use the validation loss as the early stopping loss
                # else we compute the early stopping loss using the early stopping loss computation function
                # note that early stopping loss is always computed on the validation set
                if self.compute_early_stopping_loss == self.compute_loss:
                    early_stopping_loss = val_loss
                    early_stopping_loss_batchwise = val_loss_batchwise
                else:
                    (
                        early_stopping_loss,
                        early_stopping_loss_batchwise,
                    ) = self.validate_epoch(
                        model, val_loader, epoch, self.compute_early_stopping_loss
                    )
                early_stopping_losses.append(early_stopping_loss)
                early_stopping_losses_batchwise.append(early_stopping_loss_batchwise)
                # add the early stopping loss to the metrics
                metrics["early_stopping_loss"] = early_stopping_loss
                # check if early stopping should be triggered
                if self.early_stopper.step(early_stopping_loss, model) is not None:
                    print("Early stopping triggered")
                    # self.early_stopper.step returns the best model, best train loss, best val loss, best early stopping loss
                    # and the last epoch number of training
                    (
                        model,
                        best_model_train_loss,
                        best_model_val_loss,
                        best_early_stopping_loss,
                        last_epoch,
                    ) = self.early_stop(model, train_loader, val_loader, epoch)
                    train_summary = {
                        "train_losses": train_losses,
                        "train_losses_batchwise": train_losses_batchwise,
                        "val_losses": val_losses,
                        "val_losses_batchwise": val_losses_batchwise,
                        "early_stopping_losses": early_stopping_losses,
                        "early_stopping_losses_batchwise": early_stopping_losses_batchwise,
                        "best_model_train_loss": best_model_train_loss,
                        "best_model_val_loss": best_model_val_loss,
                        "best_early_stopping_loss": best_early_stopping_loss,
                        "last_epoch": last_epoch,
                    }
                    return model, train_summary
            # if early stopping is not enabled
            # or if early stopping is enabled but early stopping is not triggered
            # just log the metrics and continue
            self.logger.log(metrics, count_phrase)

        print("Training complete.")
        last_epoch = epoch + 1
        train_summary = {
            "train_losses": train_losses,
            "train_losses_batchwise": train_losses_batchwise,
            "val_losses": val_losses,
            "val_losses_batchwise": val_losses_batchwise,
            "last_epoch": last_epoch,
        }
        # if early stopping is enabled, fetch the best model
        if self.apply_early_stopping:
            # self.early_stopper.step returns the best model, best train loss, best val loss, best early stopping loss
            # and the epoch at which early stopping was triggered
            (
                model,
                best_model_train_loss,
                best_model_val_loss,
                best_early_stopping_loss,
                last_epoch,
            ) = self.early_stop(model, train_loader, val_loader, epoch)
        else:
            # if early stopping is not enabled, set early_stopping_loss and best losses to None
            early_stopping_losses = None
            early_stopping_losses_batchwise = None
            best_model_train_loss = None
            best_model_val_loss = None
            best_early_stopping_loss = None
        train_summary["early_stopping_losses"] = early_stopping_losses
        train_summary[
            "early_stopping_losses_batchwise"
        ] = early_stopping_losses_batchwise
        train_summary["best_model_train_loss"] = best_model_train_loss
        train_summary["best_model_val_loss"] = best_model_val_loss
        train_summary["best_early_stopping_loss"] = best_early_stopping_loss
        return model, train_summary

    def train_epoch(self, model, train_loader, epoch):
        model.train()
        loss_batchwise = []
        total_loss = 0
        total_trials = 0
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            batch = [element.to(self.device) for element in batch]
            loss = self.compute_loss(model, batch, reduction="sum")

            # if has_regularizer(model):
            #     reg_value = get_reg_value(model)
            # else:
            #     reg_value = 0

            (loss / batch[0].shape[0]).backward()

            # # TODO: adding regularizer according to sysident
            # if isinstance(model, ELBOMarginal):
            #     if hasattr(
            #         model.posterior.trainable_distribution.parameter_generator[1],
            #         "regularizer",
            #     ):
            #         # detach_core=False if you want to train the core
            #         reg_value = 0
            #         for k, v in model.readout.items():
            #             reg_value += model.regularizer(
            #                 data_key=k, detach_core=self.detach_core
            #             )
            #         (loss / batch[0].shape[0] + reg_value).backward()
            #     else:
            #         (loss / batch[0].shape[0]).backward()
            # before stepping the optimizer
            # apply gradient clipping if enabled
            # or apply update skipping if enabled
            # TODO: commenting below just for debugging
            if self.apply_gradient_clipping or self.apply_update_skipping:
                # ISSUE: below is the job of the optimizer and not the trainer
                # for example, it could be that only part of the model parameters
                # are to be trained and that is known to the optimizer
                grad_norm = torch.norm(
                    torch.cat(
                        [
                            p.grad.view(-1)
                            for p in model.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                if grad_norm.item() >= self.gradient_clipping_threshold:
                    # count_phrase = (
                    #     f"Batch {batch_idx + 1}/{len(train_loader)}, Epoch {epoch + 1}"
                    # )
                    # metrics = {"grad_norm/grad_norm": grad_norm.item()}
                    # self.logger.log(metrics, count_phrase)
                    if self.apply_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=self.gradient_clipping_threshold,
                        )
                    elif self.apply_update_skipping:
                        if grad_norm >= self.update_skipping_threshold:
                            continue
            # step the optimizer
            self.optimizer.step()
            # compute the number of trials in the batch used for computing the loss
            # this computation happens only if update is not skipped
            n_trials = batch[0].shape[0]
            # update the total number of trials
            total_trials += n_trials
            # compute the mean loss for the batch
            mean_loss_batch = loss.item() / n_trials
            loss_batchwise.append(mean_loss_batch)
            # update the total loss
            total_loss += loss.item()
        # compute the mean loss for the epoch
        mean_loss_epoch = total_loss / total_trials
        return mean_loss_epoch, loss_batchwise

    # validate epoch is used for both validation and early stopping
    # hence loss_computation_fn is an argument
    def validate_epoch(self, model, val_loader, epoch, loss_computation_fn):
        model.eval()
        loss_batchwise = []
        total_loss = 0
        total_trials = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                n_trials = batch[0].shape[0]
                total_trials += n_trials
                batch = [element.to(self.device) for element in batch]
                loss = loss_computation_fn(model, batch, reduction="sum")
                # compute the mean loss for the batch
                mean_loss_batch = loss.item() / n_trials
                loss_batchwise.append(mean_loss_batch)
                # update the total loss
                total_loss += loss.item()
        # compute the mean loss for the epoch
        mean_loss_epoch = total_loss / total_trials
        return mean_loss_epoch, loss_batchwise

    def early_stop(self, model, train_loader, val_loader, epoch):
        print("Fetching best model.")
        best_model_state_dict = self.early_stopper.best_model_state_dict
        model.load_state_dict(best_model_state_dict)
        # use self.validate_epoch to compute even the train loss
        # in order to avoid gradient descent and model updates
        best_model_train_loss, _ = self.validate_epoch(
            model, train_loader, epoch, self.compute_loss
        )
        best_model_val_loss, _ = self.validate_epoch(
            model, val_loader, epoch, self.compute_loss
        )
        best_early_stopping_loss, _ = self.validate_epoch(
            model, val_loader, epoch, self.compute_early_stopping_loss
        )
        metrics = {
            "best_model_train_loss": best_model_train_loss,
            "best_model_val_loss": best_model_val_loss,
            "best_early_stopping_loss": best_early_stopping_loss,
        }
        self.logger.log(metrics, f"Epoch {epoch + 1}")
        return (
            model,
            best_model_train_loss,
            best_model_val_loss,
            best_early_stopping_loss,
            epoch + 1,
        )
