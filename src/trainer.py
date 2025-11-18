from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

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
        #self.logger.log_dict()
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                # Supervised loss
                supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
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


from copy import deepcopy
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


class MeanTeacherTrainer:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        ema_decay: float = 0.999,
        unsup_weight: float = 0.1,
        ramp_up_epochs: Optional[int] = None,  
    ):
        """
        Mean Teacher trainer for QM9.

        Args:
            models: list of models; we use models[0] as the student.
            ema_decay: EMA decay for teacher update.
            unsup_weight: MAX weight for the consistency loss on unlabeled data.
            ramp_up_epochs: if not None, linearly ramp unsup_weight from 0
                            to `unsup_weight` over this many epochs.
        """
        assert len(models) == 1, "MeanTeacherTrainer currently assumes a single student model."

        self.device = device
        self.student = models[0].to(device)
        self.teacher = deepcopy(self.student).to(device)

        # teacher is not directly optimized, only updated via EMA
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Optim / sched
        self.supervised_criterion = supervised_criterion
        self.optimizer = optimizer(params=self.student.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloaders
        self.labeled_loader = datamodule.train_dataloader()
        self.unlabeled_loader = datamodule.unsupervised_train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

        # MT hyperparams
        self.ema_decay = ema_decay
        self.unsup_weight = unsup_weight          # max unsup weight
        self.ramp_up_epochs = ramp_up_epochs      # epochs for linear ramp

    @torch.no_grad()
    def _update_teacher(self):
        alpha = self.ema_decay

        # 1) EMA on parameters (weights, biases)
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)

        # 2) Direct copy of buffers (e.g. BatchNorm running_mean / running_var)
        for t_buffer, s_buffer in zip(self.teacher.buffers(), self.student.buffers()):
            t_buffer.data.copy_(s_buffer.data)


    def _get_unlabeled_batch(self, unlabeled_iter):
        """Safely get next unlabeled batch, cycling the iterator if needed."""
        try:
            batch = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(self.unlabeled_loader)
            batch = next(unlabeled_iter)
        return batch, unlabeled_iter

    def _current_unsup_weight(self, epoch: int) -> float:
        """Linear ramp-up of unsup_weight from 0 → self.unsup_weight."""
        if self.ramp_up_epochs is None or self.ramp_up_epochs <= 0:
            return self.unsup_weight
        ramp_factor = min(1.0, epoch / float(self.ramp_up_epochs))
        return self.unsup_weight * ramp_factor

    def validate(self):
        """
        Evaluate BOTH teacher and student on the validation set.

        Returns:
            {
                "val_MSE_teacher": ...,
                "val_MSE_student": ...,
                "val_MSE": ... (alias for teacher MSE, for compatibility)
            }
        """
        self.teacher.eval()
        self.student.eval()

        teacher_losses = []
        student_losses = []

        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                teacher_preds = self.teacher(x)
                student_preds = self.student(x)

                teacher_losses.append(F.mse_loss(teacher_preds, targets).item())
                student_losses.append(F.mse_loss(student_preds, targets).item())

        val_teacher = float(np.mean(teacher_losses)) if teacher_losses else float("nan")
        val_student = float(np.mean(student_losses)) if student_losses else float("nan")

        return {
            "val_MSE_teacher": val_teacher,
            "val_MSE_student": val_student,
            "val_MSE": val_teacher,  
        }

    def train(self, total_epochs, validation_interval):
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):

            self.student.train()
            self.teacher.eval()  # teacher is used only for targets

            sup_losses_logged = []
            cons_losses_logged = []

            # compute epoch-specific unsup weight (with ramp-up)
            current_unsup_weight = self._current_unsup_weight(epoch)

            unlabeled_iter = iter(self.unlabeled_loader)

            for (x_l, y_l) in self.labeled_loader:
                x_l, y_l = x_l.to(self.device), y_l.to(self.device)

                # Get unlabeled batch (ignore its labels)
                (x_u, _), unlabeled_iter = self._get_unlabeled_batch(unlabeled_iter)
                x_u = x_u.to(self.device)

                self.optimizer.zero_grad()

                # ----- Supervised loss (student on labeled data) -----
                preds_l = self.student(x_l)
                sup_loss = self.supervised_criterion(preds_l, y_l)

                # ----- Consistency loss (student vs teacher on unlabeled) -----
                with torch.no_grad():
                    teacher_u = self.teacher(x_u)

                student_u = self.student(x_u)
                cons_loss = F.mse_loss(student_u, teacher_u)

                loss = sup_loss + current_unsup_weight * cons_loss
                loss.backward()
                self.optimizer.step()

                # EMA teacher update
                self._update_teacher()

                sup_losses_logged.append(sup_loss.detach().item())
                cons_losses_logged.append(cons_loss.detach().item())

            # Scheduler step per epoch
            self.scheduler.step()

            sup_losses_logged = float(np.mean(sup_losses_logged)) if sup_losses_logged else float("nan")
            cons_losses_logged = float(np.mean(cons_losses_logged)) if cons_losses_logged else float("nan")

            summary_dict = {
                "supervised_loss": sup_losses_logged,
                "consistency_loss": cons_losses_logged,  # unweighted cons loss
                "unsup_weight_epoch": current_unsup_weight,
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)

            self.logger.log_dict(summary_dict, step=epoch)

        final_metrics = self.validate()
        self.logger.log_dict(final_metrics, step=total_epochs)
        return list(final_metrics.values())
