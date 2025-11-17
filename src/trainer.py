from functools import partial

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
        lambda_u: float = 1.0,
    ):
        self.device = device
        self.models = models
        self.lambda_u = lambda_u

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.unsup_dataloader = datamodule.unsupervised_train_dataloader()
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

    def _consistency_loss(self, preds_unlabeled):
        """
        preds_unlabeled: list of tensors from each model on the same unlabeled batch.
        """
        if len(preds_unlabeled) <= 1:
            return torch.tensor(0.0, device=self.device)

        stacked = torch.stack(preds_unlabeled)  # [n_models, batch, out_dim]
        mean_preds = stacked.mean(dim=0, keepdim=True)
        diffs = stacked - mean_preds
        return (diffs ** 2).mean()

    def train(self, total_epochs, validation_interval):
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()

            supervised_losses_logged = []
            consistency_losses_logged = []

            # fresh iterator for unlabeled loader
            unsup_iter = iter(self.unsup_dataloader)

            for x_l, targets in self.train_dataloader:
                x_l, targets = x_l.to(self.device), targets.to(self.device)

                # get one unlabeled batch; if we hit the end, restart the unlabeled loader
                try:
                    x_u, _ = next(unsup_iter)
                except StopIteration:
                    unsup_iter = iter(self.unsup_dataloader)
                    x_u, _ = next(unsup_iter)

                x_u = x_u.to(self.device)

                self.optimizer.zero_grad()

                # supervised loss
                supervised_losses = [
                    self.supervised_criterion(model(x_l), targets)
                    for model in self.models
                ]
                supervised_loss = sum(supervised_losses) / len(self.models)

                # consistency loss
                preds_unlabeled = [model(x_u) for model in self.models]
                consistency_loss = self._consistency_loss(preds_unlabeled)

                loss = supervised_loss + self.lambda_u * consistency_loss
                loss.backward()
                self.optimizer.step()

                supervised_losses_logged.append(supervised_loss.detach().item())
                consistency_losses_logged.append(consistency_loss.detach().item())

            self.scheduler.step()

            supervised_losses_logged = float(np.mean(supervised_losses_logged))
            consistency_losses_logged = float(np.mean(consistency_losses_logged))

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
                "consistency_loss": consistency_losses_logged,
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)

            self.logger.log_dict(summary_dict, step=epoch)

        final_metrics = self.validate()
        self.logger.log_dict(final_metrics, step=total_epochs)
        return list(final_metrics.values())
