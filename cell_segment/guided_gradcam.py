import os
import torch
import matplotlib.pyplot as plt
from datasets import get_dataset
import hydra
import numpy as np
from models import get_model
from visualization.cam import GradCAM
from guided_backprop import replace_relu_with_guided_relu, guided_backprop

@hydra.main(version_base=None, config_path="config", config_name="default.yaml")
def guided_gradcam(cfg):
    # for output in right terminal
    os.chdir(hydra.utils.get_original_cwd())

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # dataset
    test_dataset, _,_ = get_dataset(cfg, split="test")
    image, label = test_dataset[1]
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

    #guided backprop
    replace_relu_with_guided_relu(model)
    gradients = guided_backprop(model, input_tensor, cfg)

    guided_cam = cam[..., np.newaxis] * gradients
    guided_cam = guided_cam - guided_cam.min()
    guided_cam = guided_cam / (guided_cam.max() + 1e-8)  
    image_tensor_squeezed = input_tensor.squeeze(0)
    visualize_guided_gradcam(guided_cam, image_tensor_squeezed)

def visualize_guided_gradcam(gradients, image_tensor):
    img_np = image_tensor.permute(1, 2, 0).cpu().detach().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    print(f"about to calculate the overlay")

    plt.figure(figsize=(6,6))
    plt.imshow(gradients)
    plt.title("Guided GradCAM")
    plt.axis("off")
    output_file_ggc = "ggc_output.png"
    plt.savefig(output_file_ggc, bbox_inches="tight", pad_inches=0)
    print(f"guided backpropagation saved as {output_file_ggc}")
    plt.show()

if __name__ == '__main__':
    guided_gradcam()
