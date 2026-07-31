from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
import os

from perovskite_segment.datasets.util import get_dataset
from perovskite_segment.models.util import get_model
from perovskite_segment.utils.image_evaluator import ImageEvaluator

@hydra.main(
    version_base=None,
    config_path="/Users/justicealuu/research_project/perovskite_segment/config",
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

    evaluator  = ImageEvaluator(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=0.90,
    )

    df = evaluator.evaluate(
        save_csv="validation_metrics.csv"
    )

    


if __name__ == "__main__":
    main()