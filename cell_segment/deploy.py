import hydra
import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from datasets import get_dataset
from models import get_model
from postprocessing.extract_cells import extract_cells


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

    extract_cells(
    model,
    test_loader,
    device,
    cfg,
)






if __name__ == "__main__":
    main()