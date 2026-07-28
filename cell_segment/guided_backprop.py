import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import get_model
from datasets import get_dataset
import hydra
import os

@hydra.main(version_base=None, config_path="config", config_name="default.yaml")
def main(cfg):
    # for output in right terminal
    os.chdir(hydra.utils.get_original_cwd())

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # dataset
    test_dataset, _, _ = get_dataset(cfg, split="test")
    image, label = test_dataset[1]
    input_tensor = image.unsqueeze(0)
    print(f"Loaded first image, label: {label}")

    # model
    ckpt_path = cfg.checkpoint_path 
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = get_model(cfg, device).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # guided backpropagation
    replace_relu_with_guided_relu(model)
    gradients = guided_backprop(model, input_tensor, cfg)
    visualize_guided_backprop(gradients, input_tensor[0])

class GuidedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0      
        grad_input[grad_output <= 0] = 0  
        return grad_input
    
def replace_relu_with_guided_relu(model):
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.forward = lambda x: GuidedReLU.apply(x)

def guided_backprop(model, input_tensor, cfg=None):
    model.eval()
    input_tensor.requires_grad = True

    # Forward
    output = model(input_tensor)

    
    if cfg.model.num_classes == 2:
        # Binary
        score = output.squeeze()

    else:
        # Ordinal (K classes → K-1 logits)
        probs = torch.sigmoid(output)
        pred_class = (probs > 0.5).sum(dim=1).item()

        idx = min(pred_class, output.shape[1] - 1)
        score = output[0, idx]  # scalar logit


    # Backward
    model.zero_grad()
    score.backward()

    # Gradients
    gradients = input_tensor.grad[0].cpu().numpy()  # [C,H,W]
    gradients = np.maximum(gradients, 0)           
    gradients = gradients / (gradients.max() + 1e-8)
    gradients = np.transpose(gradients, (1, 2, 0))  # H,W,C

    return gradients

def visualize_guided_backprop(gradients, image_tensor):
    img_np = image_tensor.permute(1, 2, 0).cpu().detach().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    print(f"about to calculate the overlay")

    plt.figure(figsize=(6,6))
    plt.imshow(gradients)
    plt.title("Guided Backpropagation")
    plt.axis("off")
    output_file_gbp = "gbp_output.png"
    plt.savefig(output_file_gbp, bbox_inches="tight", pad_inches=0)
    print(f"guided backpropagation saved as {output_file_gbp}")
    plt.show()
    


if __name__ == '__main__':
    main()
