
from itertools import chain
import hydra
import torch
from omegaconf import OmegaConf

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

    base_model_cfg = cfg.model.init

    models = []
    for _ in range(2):  # or 3 if you like long training times
        m = hydra.utils.instantiate(base_model_cfg).to(device)
        if cfg.compile_model:
            m = torch.compile(m)
        models.append(m)

    trainer = hydra.utils.instantiate(
        cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device
    )
    results = trainer.train(**cfg.trainer.train)
    results = torch.Tensor(results)



if __name__ == "__main__":
    main()
