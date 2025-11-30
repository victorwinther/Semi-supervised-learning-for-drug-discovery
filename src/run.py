import json
from pathlib import Path

import numpy as np
import math
if not hasattr(np, 'math'):
    np.math = math
from itertools import chain
import hydra
import torch
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from utils import seed_everything

@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    # print out the full config
    print(OmegaConf.to_yaml(cfg))

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    seed_everything(cfg.seed, cfg.force_deterministic)

    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)

    dm = hydra.utils.instantiate(cfg.dataset.init)

    model = hydra.utils.instantiate(cfg.model.init).to(device)

    if cfg.compile_model:
        model = torch.compile(model)
    models = [model]
    trainer = hydra.utils.instantiate(cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device,data_mean=dm.mean,data_std=dm.std)

    results = trainer.train(**cfg.trainer.train)
    if isinstance(results, torch.Tensor):
        metrics = results.detach().cpu().tolist()
        results = {f"metric_{idx}": float(value) for idx, value in enumerate(metrics)}
    elif isinstance(results, (list, tuple)):
        results = {f"metric_{idx}": float(value) for idx, value in enumerate(results)}

    # --------------------------------------------------------------
    # Save the model
    # --------------------------------------------------------------
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    best_model_path = run_dir / "model.pt"

    torch.save(
        {
            "state_dict": trainer.best_teacher_state,          # weights
            "model_config": hparams["model"]["init"],          # model init kwargs
            "data_mean": trainer.data_mean.detach().cpu(),     # for denorm
            "data_std": trainer.data_std.detach().cpu(),
        },
        best_model_path,
    )

    print(f"Saved best teacher model to {best_model_path}")

    dataset_cfg = hparams.get("dataset", {}) if isinstance(hparams, dict) else {}
    trainer_cfg = hparams.get("trainer", {}) if isinstance(hparams, dict) else {}
    model_cfg = hparams.get("model", {}) if isinstance(hparams, dict) else {}

    report = {
        "seed": int(cfg.seed),
        "trainer": trainer_cfg.get("method", "unknown"),
        "model": model_cfg.get("name", "unknown"),
        "dataset": dataset_cfg.get("dataset_name") or dataset_cfg.get("name", "unknown"),
        "metrics": results,
        "config": hparams,
    }

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2))
    print(f"Saved metrics summary to {metrics_path}")



if __name__ == "__main__":
    main()
