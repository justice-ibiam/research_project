from utils.seed import set_seed

import hydra
import torch

from datasets import get_dataset
from models import get_model

from trainer import Trainer
from evaluator import Evaluator

from losses import BCEDiceLoss


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="default.yaml"
)
def main(cfg):

    # reproducibility

    set_seed(cfg.seed)

    # device

    if torch.cuda.is_available():

        device = torch.device("cuda")

    elif torch.mps.is_available():

        device = torch.device("mps")

    else:

        device = torch.device("cpu")

    print(f"Using device: {device}")

    # datasets


    train_dataset, train_dataloader = get_dataset(
        cfg,
        split="train"
    )


    val_dataset, val_dataloader = get_dataset(
        cfg,
        split="valid"
    )

    # model

    model = get_model(
        cfg
    )

    model.to(device)

    # loss

    criterion = BCEDiceLoss()

    # evaluator

    evaluator = Evaluator(
        cfg,
        val_dataloader,
        model,
        device,
    )

    # trainer

    trainer = Trainer(
        cfg=cfg,
        train_loader=train_dataloader,
        model=model,
        evaluator=evaluator,
        criterion=criterion,
        device=device,
    )

    # training

    trainer.train()

    # final evaluation

    print("\nRunning final evaluation")

    evaluator.eval()


if __name__ == "__main__":
    main()