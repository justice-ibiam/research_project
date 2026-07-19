import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets.util import get_dataset
import hydra
import numpy as np
from models import get_model
# from diabetic_retinopathy.visualization.cam import GradCAM
from visualization.cam import GradCAM

@hydra.main(version_base=None, config_path="config", config_name="default.yaml")
def gradcam(cfg):
    # for output in right terminal
    os.chdir(hydra.utils.get_original_cwd())

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # dataset
    test_dataset, _, _ = get_dataset(cfg, split="test")
    image, label = test_dataset[10]
    print(f"Loaded first image, label: {label}")
    # denormalisation
    if isinstance(image, torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        image_denorm = image * std[:, None, None] + mean[:, None, None]
        image_denorm = image_denorm.clamp(0, 1)
        input_tensor = image.unsqueeze(0).to(device)  
    else:
        raise ValueError("Image is not a tensor!")

    # Model
    ckpt_path = cfg.checkpoint_path 
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = get_model(cfg, device).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    grad_cam = GradCAM(cfg, model, device, input_tensor)
    cam, pred_value = grad_cam.compute_cam()


    img_np = image_denorm.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    plt.imshow(img_np)
    plt.imshow(cam, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.title(f"Grad-CAM: output: {pred_value} , label: {label}")

    # save output
    output_file = "gradcam_output.png"
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    print(f"Grad-CAM saved as {output_file}")

    plt.show()

if __name__ == '__main__':
    gradcam()
