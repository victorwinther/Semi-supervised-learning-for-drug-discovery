from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

    def _evaluate_loader(self, dataloader):
        for model in self.models:
            model.eval()

        losses = []
        with torch.no_grad():
            for x, targets in dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)
                loss = torch.nn.functional.mse_loss(avg_preds, targets)
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    def validate(self):
        return {"val_MSE": self._evaluate_loader(self.val_dataloader)}

    def test(self):
        return {"test_MSE": self._evaluate_loader(self.test_dataloader)}

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
        test_metrics = self.test()
        summary = {**final_metrics, **test_metrics}
        self.logger.log_dict(summary, step=total_epochs)
        return summary


from copy import deepcopy
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from typing import Optional, List
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
        consistency_criterion: str = "mse",
        augment_coords: bool = False,
        coord_noise_std: float = 0.05,
        use_amp: bool = False,
        data_mean=None,
        data_std=None,
    ):

        assert (
            len(models) == 1
        ), "MeanTeacherTrainer currently assumes a single student model."

        self.device = device
        self.student = models[0].to(device)

        # Initialize teacher after student is on device
        self.teacher = self._create_teacher(self.student)

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
        self.unsup_weight = unsup_weight
        self.ramp_up_epochs = ramp_up_epochs
        self.consistency_criterion = consistency_criterion

        # Augmentation settings
        self.augment_coords = augment_coords
        self.coord_noise_std = coord_noise_std

        # Mixed Precision Setup
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Track global step for more granular EMA updates
        self.global_step = 0

        # Store normalization stats on device
        # If None provided, default to no-op (mean=0, std=1)
        self.data_mean = (
            data_mean.to(device)
            if data_mean is not None
            else torch.tensor(0.0, device=device)
        )
        self.data_std = (
            data_std.to(device)
            if data_std is not None
            else torch.tensor(1.0, device=device)
        )

        # Best-checkpoint tracking
        self.best_val = float("inf")
        self.best_teacher_state = deepcopy(self.teacher.state_dict())
        self.best_student_state = deepcopy(self.student.state_dict())

    def _create_teacher(self, student):
        """Create teacher model as a deep copy of student."""
        teacher = deepcopy(student).to(self.device)

        # Teacher is not directly optimized, only updated via EMA
        for p in teacher.parameters():
            p.requires_grad = False

        teacher.eval()  # Teacher always in eval mode
        return teacher

    def _augment_batch(self, batch):
        """Apply coordinate noise augmentation to molecular graphs."""
        if not self.augment_coords:
            return batch

        # For PyG Data objects, we need to clone the coordinate tensor
        if hasattr(batch, "pos") and batch.pos is not None:
            # Shallow copy batch structure, deep copy pos
            import copy

            batch = copy.copy(batch)
            noise = torch.randn_like(batch.pos) * self.coord_noise_std
            batch.pos = batch.pos + noise

        return batch

    @torch.no_grad()
    def _update_teacher(self):
        """Update teacher using EMA on both parameters and buffers."""
        alpha = self.ema_decay

        # EMA on parameters (weights, biases)
        for t_param, s_param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)

        # EMA on buffers (BatchNorm running stats, etc.)
        for t_buffer, s_buffer in zip(self.teacher.buffers(), self.student.buffers()):
            t_buffer.data.mul_(alpha).add_(s_buffer.data, alpha=1.0 - alpha)

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

    def _compute_consistency_loss(self, student_pred, teacher_pred):
        """Compute consistency loss between student and teacher predictions."""
        if self.consistency_criterion == "mse":
            return F.mse_loss(student_pred, teacher_pred)
        elif self.consistency_criterion == "l1":
            return F.l1_loss(student_pred, teacher_pred)
        else:
            raise ValueError(
                f"Unknown consistency criterion: {self.consistency_criterion}"
            )

    def _evaluate_loader(self, dataloader, prefix: str):
        """Evaluate teacher and student on the provided dataloader."""
        self.teacher.eval()
        self.student.eval()

        teacher_losses_real = []
        student_losses_real = []
        teacher_losses_norm = []
        student_losses_norm = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, targets = batch
                else:
                    x, targets = batch[0], batch[1]

                x, targets = x.to(self.device), targets.to(self.device)

                # 1. Get predictions (These are normalized values)
                teacher_preds_norm = self.teacher(x)
                student_preds_norm = self.student(x)

                # Targets normalized to the same space as training
                targets_norm = (targets - self.data_mean) / self.data_std

                # 2. Denormalize to Real Units (e.g. eV) before computing metrics
                teacher_preds_real = (
                    teacher_preds_norm * self.data_std
                ) + self.data_mean
                student_preds_real = (
                    student_preds_norm * self.data_std
                ) + self.data_mean

                # 3. Compute error against real targets
                teacher_losses_real.append(
                    F.mse_loss(teacher_preds_real, targets).item()
                )
                student_losses_real.append(
                    F.mse_loss(student_preds_real, targets).item()
                )

                teacher_losses_norm.append(
                    F.mse_loss(teacher_preds_norm, targets_norm).item()
                )
                student_losses_norm.append(
                    F.mse_loss(student_preds_norm, targets_norm).item()
                )

        teacher_score_real = (
            float(np.mean(teacher_losses_real)) if teacher_losses_real else float("nan")
        )
        student_score_real = (
            float(np.mean(student_losses_real)) if student_losses_real else float("nan")
        )
        teacher_score_norm = (
            float(np.mean(teacher_losses_norm)) if teacher_losses_norm else float("nan")
        )
        student_score_norm = (
            float(np.mean(student_losses_norm)) if student_losses_norm else float("nan")
        )

        return {
            f"{prefix}_MSE_teacher": teacher_score_real,
            f"{prefix}_MSE_student": student_score_real,
            f"{prefix}_MSE": teacher_score_real,
            f"{prefix}_MSE_norm_teacher": teacher_score_norm,  # same space as supervised_loss
            f"{prefix}_MSE_norm_student": student_score_norm,
        }

    def validate(self):
        return self._evaluate_loader(self.val_dataloader, "val")

    def test(self):
        return self._evaluate_loader(self.test_dataloader, "test")

    def evaluate_train(self):
        """Evaluate on the labeled training loader in real units."""
        return self._evaluate_loader(self.labeled_loader, "train")

    def train(self, total_epochs, validation_interval=1):
        """
        Train the Mean Teacher model with Mixed Precision support.
        """

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            self.student.train()
            self.teacher.eval()

            sup_losses_logged = []
            cons_losses_logged = []
            total_losses_logged = []

            current_unsup_weight = self._current_unsup_weight(epoch)
            unlabeled_iter = iter(self.unlabeled_loader)

            for batch_idx, batch in enumerate(self.labeled_loader):
                # 1. Prepare Data
                if len(batch) == 2:
                    x_l, y_l = batch
                else:
                    x_l, y_l = batch[0], batch[1]

                x_l, y_l = x_l.to(self.device), y_l.to(self.device)

                # Normalize Targets to N(0,1) for training stability
                y_l_normalized = (y_l - self.data_mean) / self.data_std

                unlabeled_batch, unlabeled_iter = self._get_unlabeled_batch(
                    unlabeled_iter
                )
                if len(unlabeled_batch) == 2:
                    x_u, _ = unlabeled_batch
                else:
                    x_u = unlabeled_batch[0]
                x_u = x_u.to(self.device)

                self.optimizer.zero_grad()

                # 2. Forward Pass with Autocast (Mixed Precision)
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.use_amp
                ):
                    # Supervised forward
                    preds_l = self.student(x_l)
                    sup_loss = self.supervised_criterion(preds_l, y_l_normalized)

                    # Consistency forward
                    x_u_student = self._augment_batch(x_u)
                    x_u_teacher = x_u

                    with torch.no_grad():
                        teacher_u = self.teacher(x_u_teacher)

                    student_u = self.student(x_u_student)
                    cons_loss = self._compute_consistency_loss(student_u, teacher_u)

                    # Total loss
                    total_loss = sup_loss + current_unsup_weight * cons_loss

                # 3. Backward Pass with Scaler
                self.scaler.scale(total_loss).backward()

                # Unscale logic for Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=10.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # 4. Update Teacher and Log
                self._update_teacher()
                self.global_step += 1

                sup_losses_logged.append(sup_loss.detach().item())
                cons_losses_logged.append(cons_loss.detach().item())
                total_losses_logged.append(total_loss.detach().item())

            # Stats
            avg_sup_loss = (
                float(np.mean(sup_losses_logged)) if sup_losses_logged else float("nan")
            )
            avg_cons_loss = (
                float(np.mean(cons_losses_logged))
                if cons_losses_logged
                else float("nan")
            )
            avg_total_loss = (
                float(np.mean(total_losses_logged))
                if total_losses_logged
                else float("nan")
            )

            summary_dict = {
                "epoch": epoch,
                "supervised_loss": avg_sup_loss,
                "consistency_loss": avg_cons_loss,
                "total_loss": avg_total_loss,
                "unsup_weight": current_unsup_weight,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            # --- Validation Logic ---
            val_metrics = {}
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(
                    {
                        "sup": f"{avg_sup_loss:.4f}",
                        "cons": f"{avg_cons_loss:.4f}",
                        "val_t": f"{val_metrics.get('val_MSE_teacher', float('nan')):.4f}",
                    }
                )

                # --- Best checkpoint tracking ---
                current_val = val_metrics.get("val_MSE_teacher", float("inf"))
                if current_val < self.best_val:
                    self.best_val = current_val
                    self.best_teacher_state = deepcopy(self.teacher.state_dict())
                    self.best_student_state = deepcopy(self.student.state_dict())

            # --- SCHEDULER LOGIC (MODIFIED) ---
            # Check if using ReduceLROnPlateau
            if isinstance(self.scheduler, ReduceLROnPlateau):
                # Only step if we actually validated this epoch and have a metric
                if "val_MSE_teacher" in val_metrics:
                    self.scheduler.step(val_metrics["val_MSE_teacher"])
            else:
                # Standard schedulers (StepLR, Cosine, etc.) step every epoch
                self.scheduler.step()

            self.logger.log_dict(summary_dict, step=epoch)

        # Load best checkpoint before final evaluation
        self.teacher.load_state_dict(self.best_teacher_state)
        self.student.load_state_dict(self.best_student_state)

        # Final evaluation (best checkpoint)
        final_val_metrics = self.validate()
        test_metrics = self.test()
        train_metrics = self.evaluate_train()

        final_summary = {
            "final_epoch": total_epochs,
            **{f"final_{k}": v for k, v in final_val_metrics.items()},
            **test_metrics,
            **train_metrics,
        }

        self.logger.log_dict(final_summary, step=total_epochs)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(
            f"Final Val MSE (Teacher): {final_val_metrics.get('val_MSE_teacher', float('nan')):.6f}"
        )
        print("=" * 60)

        return final_summary


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
        if (
            self.unsupervised_dataloader is not None
            and len(self.unsupervised_dataloader) == 0
        ):
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
        other_preds = [
            pred for idx, pred in enumerate(teacher_preds) if idx != exclude_idx
        ]
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
                pseudo = (
                    torch.nn.functional.one_hot(hard, num_classes=num_classes)
                    .float()
                    .mean(dim=0)
                )
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

    def _evaluate_loader(self, dataloader):
        for model in self.models:
            model.eval()

        losses = []
        with torch.no_grad():
            for x, targets in dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                preds = torch.stack([model(x) for model in self.models]).mean(dim=0)
                loss = torch.nn.functional.mse_loss(preds, targets)
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    def validate(self):
        return {"val_MSE": self._evaluate_loader(self.val_dataloader)}

    def test(self):
        return {"test_MSE": self._evaluate_loader(self.test_dataloader)}

    def train(self, total_epochs, validation_interval):
        unsupervised_iterator = (
            iter(self.unsupervised_dataloader)
            if self.unsupervised_dataloader is not None
            else None
        )

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()

            supervised_losses_logged = []
            unsupervised_losses_logged = []
            epoch_unsupervised_weight = self._consistency_weight(epoch)

            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                unsupervised_batch, unsupervised_iterator = (
                    self._get_unsupervised_batch(unsupervised_iterator)
                )

                self.optimizer.zero_grad()
                supervised_losses = [
                    self.supervised_criterion(model(x), targets)
                    for model in self.models
                ]
                supervised_loss = torch.stack(supervised_losses).mean()
                unsupervised_loss = self._cross_pseudo_loss(
                    unsupervised_batch, epoch_unsupervised_weight
                )
                (supervised_loss + unsupervised_loss).backward()
                self.optimizer.step()

                supervised_losses_logged.append(supervised_loss.detach().item())
                if unsupervised_loss.item() > 0:
                    unsupervised_losses_logged.append(unsupervised_loss.detach().item())

            self.scheduler.step()

            summary_dict = {
                "supervised_loss": (
                    float(np.mean(supervised_losses_logged))
                    if supervised_losses_logged
                    else 0.0
                ),
                "unsupervised_loss": (
                    float(np.mean(unsupervised_losses_logged))
                    if unsupervised_losses_logged
                    else 0.0
                ),
                "unsupervised_weight": epoch_unsupervised_weight,
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)

        final_metrics = self.validate()
        test_metrics = self.test()
        summary = {**final_metrics, **test_metrics}
        self.logger.log_dict(summary, step=total_epochs)
        return summary


class ConsistencyAugmentationTrainer:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        consistency_weight: float = 1.0,
        ramp_up_epochs: Optional[int] = None,
        aug_noise_std: float = 0.02,  # Std dev for coordinate noise
        aug_mask_prob: float = 0.1,  # Probability of masking node features
    ):

        self.device = device
        self.model = models[0].to(device)  # Single model approach

        # Optim / sched
        self.supervised_criterion = supervised_criterion
        self.optimizer = optimizer(params=self.model.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloaders
        self.labeled_loader = datamodule.train_dataloader()
        self.unlabeled_loader = datamodule.unsupervised_train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

        # Hyperparams
        self.consistency_weight = consistency_weight
        self.ramp_up_epochs = ramp_up_epochs
        self.aug_noise_std = aug_noise_std
        self.aug_mask_prob = aug_mask_prob

    def _get_unlabeled_batch(self, unlabeled_iter):
        """Safely get next unlabeled batch, cycling the iterator if needed."""
        try:
            batch = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(self.unlabeled_loader)
            batch = next(unlabeled_iter)
        return batch, unlabeled_iter

    def _current_consistency_weight(self, epoch: int) -> float:
        """Linear ramp-up."""
        if self.ramp_up_epochs is None or self.ramp_up_epochs <= 0:
            return self.consistency_weight
        ramp_factor = min(1.0, epoch / float(self.ramp_up_epochs))
        return self.consistency_weight * ramp_factor

    def augment_batch(self, batch):
        """
        Applies non-destructive augmentations to a batch of molecules.
        """
        # Clone to ensure we don't mess up the original batch for the first pass
        batch_aug = batch.clone()

        # 1. Coordinate Noise (Crucial for 3D models: SchNet, DimeNet)
        if hasattr(batch_aug, "pos") and batch_aug.pos is not None:
            noise = torch.randn_like(batch_aug.pos) * self.aug_noise_std
            batch_aug.pos = batch_aug.pos + noise

        # 2. Feature Masking (Crucial for 2D models: GCN, GIN)
        if hasattr(batch_aug, "x") and batch_aug.x is not None:
            mask = (
                torch.rand(batch_aug.x.size(), device=self.device) < self.aug_mask_prob
            )
            batch_aug.x[mask] = 0.0

        return batch_aug

    def _evaluate_loader(self, dataloader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for x, targets in dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                preds = self.model(x)
                losses.append(F.mse_loss(preds, targets).item())
        return float(np.mean(losses)) if losses else float("nan")

    def validate(self):
        return {"val_MSE": self._evaluate_loader(self.val_dataloader)}

    def test(self):
        return {"test_MSE": self._evaluate_loader(self.test_dataloader)}

    def train(self, total_epochs, validation_interval):
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):

            self.model.train()

            sup_losses_logged = []
            cons_losses_logged = []

            current_weight = self._current_consistency_weight(epoch)
            unlabeled_iter = iter(self.unlabeled_loader)

            for x_l, y_l in self.labeled_loader:
                x_l, y_l = x_l.to(self.device), y_l.to(self.device)

                # Get unlabeled batch
                (x_u, _), unlabeled_iter = self._get_unlabeled_batch(unlabeled_iter)
                x_u = x_u.to(self.device)

                # Create Augmented version of unlabeled batch
                x_u_aug = self.augment_batch(x_u)

                self.optimizer.zero_grad()

                # 1. Supervised Loss
                preds_l = self.model(x_l)
                sup_loss = self.supervised_criterion(preds_l, y_l)

                # 2. Consistency Loss (Augmentation)
                # Get prediction on original Clean data
                pred_u_clean = self.model(x_u)

                # Get prediction on Augmented data
                pred_u_aug = self.model(x_u_aug)

                # Minimize difference.
                # We detach 'clean' targets to stop gradient flowing back through them
                # (Standard practice in UDA/Consistency Regularization)
                cons_loss = F.mse_loss(pred_u_aug, pred_u_clean.detach())

                loss = sup_loss + (current_weight * cons_loss)

                loss.backward()
                self.optimizer.step()

                sup_losses_logged.append(sup_loss.detach().item())
                cons_losses_logged.append(cons_loss.detach().item())

            self.scheduler.step()

            sup_losses_logged = np.mean(sup_losses_logged)
            cons_losses_logged = np.mean(cons_losses_logged)

            summary_dict = {
                "supervised_loss": sup_losses_logged,
                "consistency_loss": cons_losses_logged,
                "weight": current_weight,
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)

            self.logger.log_dict(summary_dict, step=epoch)

        final_metrics = self.validate()
        test_metrics = self.test()
        summary = {**final_metrics, **test_metrics}
        self.logger.log_dict(summary, step=total_epochs)
        return summary
