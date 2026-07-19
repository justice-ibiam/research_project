import hydra
import torch
import os

from datasets import get_dataset
from models import get_model
from evaluator import Evaluator
from utils.visualize_predictions import visualize_predictions


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="default.yaml"
)
def main(cfg):

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    model = get_model(
        cfg
    )

    if os.path.exists(
        cfg.checkpoint_path
    ):

        checkpoint = torch.load(
            cfg.checkpoint_path,
            map_location=device,
            weights_only=False
        )
        

        model.load_state_dict(
            checkpoint["model_state"]
        )
        model.to(device)
   

        print(
            "Checkpoint loaded"
        )
        print(next(model.parameters()).device)

    _, test_loader = (
        get_dataset(
            cfg,
            split="test"
        )
    )

    evaluator = Evaluator(
        cfg,
        test_loader,
        model,
        device
    )

    evaluator.eval()

    visualize_predictions(
        cfg,
        model,
        test_loader,
        device,
        num_images=71
    )


if __name__ == "__main__":
    main()