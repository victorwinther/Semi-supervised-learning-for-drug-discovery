from functools import partial
from copy import deepcopy
from typing import Optional, Tuple

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
        unsupervised_weight: float = 1.0,
        pseudo_label_std_threshold: Optional[float] = 0.15,
        ema_decay: float = 0.99,
        consistency_rampup_epochs: int = 50,
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
        self.unsupervised_dataloader = (
            datamodule.unsupervised_train_dataloader()
            if hasattr(datamodule, "unsupervised_train_dataloader")
            else None
        )
        if self.unsupervised_dataloader is not None and len(self.unsupervised_dataloader) == 0:
            self.unsupervised_dataloader = None

        # Logging
        self.logger = logger
        self.global_step = 0

        # Semi-supervised hyper-parameters
        self.unsupervised_weight = unsupervised_weight
        self.pseudo_label_std_threshold = pseudo_label_std_threshold
        self.ema_decay = ema_decay
        self.consistency_rampup_epochs = consistency_rampup_epochs

        # Teacher models (EMA of students) for stable pseudo labels
        self.teacher_models = [deepcopy(model).to(self.device).eval() for model in self.models]
        for teacher in self.teacher_models:
            for param in teacher.parameters():
                param.requires_grad_(False)

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

    def _update_teacher_models(self):
        if not self.teacher_models:
            return
        for teacher, student in zip(self.teacher_models, self.models):
            for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
                teacher_param.data.mul_(self.ema_decay)
                teacher_param.data.add_((1 - self.ema_decay) * student_param.data)

    def _consistency_weight(self, epoch: int) -> float:
        if self.unsupervised_weight <= 0 or self.consistency_rampup_epochs <= 0:
            return max(0.0, self.unsupervised_weight)
        ramp = min(1.0, epoch / float(self.consistency_rampup_epochs))
        return self.unsupervised_weight * ramp

    def _get_unsupervised_batch(self, iterator):
        if iterator is None:
            return None, iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(self.unsupervised_dataloader)
            batch = next(iterator)
        return batch, iterator

    def _pseudo_label_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_u, _ = batch
        x_u = x_u.to(self.device)
        with torch.no_grad():
            teacher_preds = torch.stack([teacher(x_u) for teacher in self.teacher_models])
            pseudo_targets = teacher_preds.mean(dim=0)
            disagreement = teacher_preds.std(dim=0)
            if disagreement.ndim > 1:
                disagreement = disagreement.mean(dim=-1, keepdim=True)

        sample_disagreement = disagreement.view(disagreement.size(0))
        if self.pseudo_label_std_threshold is not None:
            mask = sample_disagreement <= self.pseudo_label_std_threshold
        else:
            mask = torch.ones_like(sample_disagreement, dtype=torch.bool, device=sample_disagreement.device)
        return x_u, pseudo_targets.detach(), mask

    def _compute_unsupervised_loss(self, batch, weight: float):
        if (
            batch is None
            or self.unsupervised_dataloader is None
            or weight <= 0
        ):
            return torch.tensor(0.0, device=self.device), 0.0

        x_u, pseudo_targets, mask = self._pseudo_label_batch(batch)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device), 0.0

        pseudo_targets_masked = pseudo_targets[mask]
        unsupervised_losses = []
        for model in self.models:
            preds = model(x_u)[mask]
            unsupervised_losses.append(self.supervised_criterion(preds, pseudo_targets_masked))
        unsupervised_loss = sum(unsupervised_losses) / len(self.models)
        acceptance_rate = mask.float().mean().item()
        return unsupervised_loss * weight, acceptance_rate

    def train(self, total_epochs, validation_interval):
        #self.logger.log_dict()
        unsupervised_iterator = iter(self.unsupervised_dataloader) if self.unsupervised_dataloader is not None else None
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            unsupervised_losses_logged = []
            pseudo_acceptance_logged = []
            epoch_unsupervised_weight = self._consistency_weight(epoch)

            for x, targets in self.train_dataloader:
                unsupervised_batch, unsupervised_iterator = self._get_unsupervised_batch(unsupervised_iterator)
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                # Supervised loss
                supervised_losses = [
                    self.supervised_criterion(model(x), targets)
                    for model in self.models
                ]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))  # type: ignore
                unsupervised_loss, acceptance_rate = self._compute_unsupervised_loss(
                    unsupervised_batch, epoch_unsupervised_weight
                )
                loss = supervised_loss + unsupervised_loss
                loss.backward()  # type: ignore
                self.optimizer.step()
                self._update_teacher_models()
                self.global_step += 1
                if unsupervised_loss.item() > 0:
                    unsupervised_losses_logged.append(unsupervised_loss.detach().item())
                    pseudo_acceptance_logged.append(acceptance_rate)

            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)
            unsupervised_losses_logged = np.mean(unsupervised_losses_logged) if unsupervised_losses_logged else 0.0
            pseudo_acceptance_logged = np.mean(pseudo_acceptance_logged) if pseudo_acceptance_logged else 0.0

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
                "unsupervised_loss": unsupervised_losses_logged,
                "pseudo_acceptance_rate": pseudo_acceptance_logged,
                "unsupervised_weight": epoch_unsupervised_weight,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)

        final_metrics = self.validate()
        self.logger.log_dict(final_metrics, step=total_epochs)
        return list(final_metrics.values())


class NCPS:
    """
    n-CPS style trainer from https://arxiv.org/abs/2112.07528.
    Multiple students are trained jointly: each uses the averaged predictions
    of the other networks as pseudo-labels on unlabeled data.
    """

    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        num_networks: int = 1,  # unused, kept for Hydra compatibility
        unsupervised_criterion=None,
        unsupervised_weight: float = 1.0,
        consistency_rampup_epochs: int = 50,
        hard_pseudo_labels: bool = False,
        num_classes: Optional[int] = None,
    ):
        self.device = device
        self.models = [model.to(device) for model in models]
        assert len(self.models) >= 1, "NCPS requires at least one model instance."

        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion or supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        self.unsupervised_dataloader = (
            datamodule.unsupervised_train_dataloader()
            if hasattr(datamodule, "unsupervised_train_dataloader")
            else None
        )
        if self.unsupervised_dataloader is not None and len(self.unsupervised_dataloader) == 0:
            self.unsupervised_dataloader = None

        self.logger = logger
        self.unsupervised_weight = unsupervised_weight
        self.consistency_rampup_epochs = consistency_rampup_epochs
        self.hard_pseudo_labels = hard_pseudo_labels
        self.num_classes = num_classes

    def _consistency_weight(self, epoch: int) -> float:
        if self.unsupervised_weight <= 0 or self.consistency_rampup_epochs <= 0:
            return max(0.0, self.unsupervised_weight)
        ramp = min(1.0, epoch / float(self.consistency_rampup_epochs))
        return self.unsupervised_weight * ramp

    def _get_unsupervised_batch(self, iterator):
        if iterator is None:
            return None, iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(self.unsupervised_dataloader)
            batch = next(iterator)
        return batch, iterator

    def _build_pseudo_target(self, teacher_preds, exclude_idx: int):
        if len(self.models) <= 1:
            return None
        other_preds = [pred for idx, pred in enumerate(teacher_preds) if idx != exclude_idx]
        if not other_preds:
            return None
        stacked = torch.stack(other_preds)
        if self.hard_pseudo_labels:
            if stacked.dim() <= 2:
                # Binary/1D case -> sign disagreement threshold at zero.
                pseudo = (stacked.mean(dim=0) > 0).float()
            else:
                num_classes = self.num_classes or stacked.size(-1)
                hard = stacked.argmax(dim=-1)
                pseudo = torch.nn.functional.one_hot(hard, num_classes=num_classes).float().mean(dim=0)
        else:
            pseudo = stacked.mean(dim=0)
        return pseudo.detach()

    def _cross_pseudo_loss(self, batch, weight: float):
        if (
            batch is None
            or self.unsupervised_dataloader is None
            or weight <= 0
            or len(self.models) <= 1
        ):
            return torch.tensor(0.0, device=self.device)

        x_u, _ = batch
        x_u = x_u.to(self.device)
        with torch.no_grad():
            teacher_preds = [model(x_u) for model in self.models]

        losses = []
        for idx, student in enumerate(self.models):
            pseudo_target = self._build_pseudo_target(teacher_preds, exclude_idx=idx)
            if pseudo_target is None:
                continue
            student_pred = student(x_u)
            losses.append(self.unsupervised_criterion(student_pred, pseudo_target))
        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean() * weight

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                preds = torch.stack([model(x) for model in self.models]).mean(dim=0)
                val_loss = torch.nn.functional.mse_loss(preds, targets)
                val_losses.append(val_loss.item())
        return {"val_MSE": float(np.mean(val_losses)) if val_losses else float("nan")}

    def train(self, total_epochs, validation_interval):
        unsupervised_iterator = (
            iter(self.unsupervised_dataloader) if self.unsupervised_dataloader is not None else None
        )

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()

            supervised_losses_logged = []
            unsupervised_losses_logged = []
            epoch_unsupervised_weight = self._consistency_weight(epoch)

            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                unsupervised_batch, unsupervised_iterator = self._get_unsupervised_batch(unsupervised_iterator)

                self.optimizer.zero_grad()
                supervised_losses = [
                    self.supervised_criterion(model(x), targets) for model in self.models
                ]
                supervised_loss = torch.stack(supervised_losses).mean()
                unsupervised_loss = self._cross_pseudo_loss(unsupervised_batch, epoch_unsupervised_weight)
                (supervised_loss + unsupervised_loss).backward()
                self.optimizer.step()

                supervised_losses_logged.append(supervised_loss.detach().item())
                if unsupervised_loss.item() > 0:
                    unsupervised_losses_logged.append(unsupervised_loss.detach().item())

            self.scheduler.step()

            summary_dict = {
                "supervised_loss": float(np.mean(supervised_losses_logged)) if supervised_losses_logged else 0.0,
                "unsupervised_loss": float(np.mean(unsupervised_losses_logged)) if unsupervised_losses_logged else 0.0,
                "unsupervised_weight": epoch_unsupervised_weight,
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
