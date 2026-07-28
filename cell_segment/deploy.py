import hydra
import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from datasets import get_dataset
from models import get_model
from evaluator import Evaluator
from utils.visualize_predictions import visualize_predictions
from utils.perspective_rectification import find_panel_contour
from utils.perspective_rectification import get_panel_corners
from utils.perspective_rectification import order_points
from utils.perspective_rectification import rectify_image


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

        image = cv2.imread(
            str(cfg.image_path)
        )

        original_image = image.copy()

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
        image = cv2.resize(
            image,
            (320,240),
            interpolation=cv2.INTER_LINEAR
        )
        # convert image
        # HWC -> CHW
        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        image = image.permute(2, 0, 1)
        image = image / 255.0
        image = image.unsqueeze(0)

        # Move image to device
        image = image.to(device)

        # Move model to device (do this once, not every inference)
        model = model.to(device)

        with torch.no_grad():
            output = model(image)

        # Convert logits to probabilities
        output = torch.sigmoid(output)

        # Threshold to binary mask
        output = (output > 0.5).float()

        # Remove batch and channel dimensions
        output = output.squeeze()

        # Move to CPU and convert to NumPy
        output = output.cpu().numpy().astype(np.uint8)
        contour = find_panel_contour(output)
        corners = get_panel_corners(contour)
        corners = order_points(corners)

        print(corners)

        sx = original_image.shape[1] / 320
        sy = original_image.shape[0] / 240

        corners[:, 0] *= sx
        corners[:, 1] *= sy

        print(corners)

        rectify_image(original_image, corners)





    # _, test_loader = (
    #     get_dataset(
    #         cfg,
    #         split="test"
    #     )
    # )

    # evaluator = Evaluator(
    #     cfg,
    #     test_loader,
    #     model,
    #     device
    # )

    # evaluator.eval()

    # visualize_predictions(
    #     model,
    #     test_loader,
    #     device,
    #     num_images=10
    # )


if __name__ == "__main__":
    main()