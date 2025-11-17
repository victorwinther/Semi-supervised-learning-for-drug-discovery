from functools import partial
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm


class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
    ):
        self.device = device
        self.models = models

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []

        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                # Ensemble prediction
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)

                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs, validation_interval):
        # self.logger.log_dict()
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                # Supervised loss
                supervised_losses = [
                    self.supervised_criterion(model(x), targets)
                    for model in self.models
                ]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))  # type: ignore
                loss = supervised_loss
                loss.backward()  # type: ignore
                self.optimizer.step()
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)

        final_metrics = self.validate()
        self.logger.log_dict(final_metrics, step=total_epochs)
        return list(final_metrics.values())


class MeanTeacher:
    """
    Simple semi-supervised trainer that combines:
    - Mean Teacher: an EMA teacher network provides stable pseudo-labels.
    - Consistency regularization: the student is encouraged to match the teacher on unlabeled data.
    """

    def __init__(
        self,
        supervised_criterion,
        consistency_weight: float,
        ema_decay: float,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
    ):
        self.device = device
        assert len(models) == 1, "MeanTeacher expects a single student model."
        self.student = models[0].to(device)

        # Teacher is an EMA copy; no gradients are tracked for it.
        self.teacher = deepcopy(self.student).to(device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Optimizer/scheduler track only the student.
        self.supervised_criterion = supervised_criterion
        self.consistency_weight = consistency_weight
        self.ema_decay = ema_decay
        self.optimizer = optimizer(params=self.student.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloaders
        self.sup_loader = datamodule.train_dataloader()
        self.unsup_loader = datamodule.unsupervised_train_dataloader()
        self.val_loader = datamodule.val_dataloader()

        self.logger = logger

    @torch.no_grad()
    def _update_teacher(self):
        # EMA update: teacher = ema * teacher + (1-ema) * student
        for t_param, s_param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            t_param.data.mul_(self.ema_decay).add_(
                s_param.data, alpha=1 - self.ema_decay
            )

    def _validate_teacher(self):
        self.teacher.eval()
        losses = []
        with torch.no_grad():
            for batch, targets in self.val_loader:
                batch, targets = batch.to(self.device), targets.to(self.device)
                preds = self.teacher(batch)
                loss = torch.nn.functional.mse_loss(preds, targets)
                losses.append(loss.item())
        return {"val_MSE": float(np.mean(losses)) if losses else float("nan")}

    def train(self, total_epochs: int, validation_interval: int):
        unsup_iter = iter(self.unsup_loader)

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            self.student.train()
            sup_losses, unsup_losses = [], []

            for sup_batch, sup_targets in self.sup_loader:
                sup_batch, sup_targets = sup_batch.to(self.device), sup_targets.to(
                    self.device
                )
                try:
                    unsup_batch, _ = next(unsup_iter)
                except StopIteration:
                    unsup_iter = iter(self.unsup_loader)
                    unsup_batch, _ = next(unsup_iter)
                unsup_batch = unsup_batch.to(self.device)

                self.optimizer.zero_grad()

                # Supervised loss on labeled data
                sup_preds = self.student(sup_batch)
                sup_loss = self.supervised_criterion(sup_preds, sup_targets)

                # Teacher prediction (no grad)
                with torch.no_grad():
                    teacher_preds = self.teacher(unsup_batch).detach()

                # Student prediction on unlabeled data
                stud_pred = self.student(unsup_batch)
                consistency_loss = torch.nn.functional.mse_loss(
                    stud_pred, teacher_preds
                )

                loss = sup_loss + self.consistency_weight * consistency_loss
                loss.backward()
                self.optimizer.step()

                sup_losses.append(sup_loss.item())
                unsup_losses.append(consistency_loss.item())

                # EMA update after each step
                self._update_teacher()

            self.scheduler.step()

            summary = {
                "loss_supervised": float(np.mean(sup_losses)) if sup_losses else 0.0,
                "loss_consistency": (
                    float(np.mean(unsup_losses)) if unsup_losses else 0.0
                ),
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self._validate_teacher()
                summary.update(val_metrics)
                pbar.set_postfix(summary)

            self.logger.log_dict(summary, step=epoch)

        final_metrics = self._validate_teacher()
        self.logger.log_dict(final_metrics, step=total_epochs)
        return list(final_metrics.values())
