import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import get_dataset

class GradCAM:
    def __init__(self, cfg, model, device, input_tensor):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.input_tensor = input_tensor.to(device)
        self.activations = None
        self.gradients = None
        self.pred_value = None
    
    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute_cam(self):
        if self.cfg.model.name in ['BasicCNN', 'own_cnn']:
            target_layer = self.model.conv_layers[-1].conv_2
        elif self.cfg.model.name == 'resnet50':
            target_layer = self.model.backbone.layer4[-1].conv3
        elif self.cfg.model.name == 'densenet121':
            target_layer = self.model.backbone.features.denseblock4
        elif self.cfg.model.name == 'efficientnet_b0':
            target_layer = self.model.features[-1]
        else:
            raise ValueError(f'Model "{self.cfg.model.name}" unknown')

        # hooks
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

        # forward pass 
        output = self.model(self.input_tensor)
        print("Output shape:", output.shape)

        # prediction & target selection
        if self.cfg.model.num_classes == 2:
            # BINARY 
            prob = torch.sigmoid(output)
            pred_class = int(prob.item() > 0.5)
            score = output.squeeze()

        else:
            # MULTICLASS 
            probs = torch.sigmoid(output)
            pred_class =(probs > 0.5).sum(dim=1).item()
            idx = min(pred_class, output.shape[1] - 1)
            score = output[0, idx]   # scalar logit for predicted class

        print("Predicted class:", pred_class)

        # backward pass 
        self.model.zero_grad()
        score.backward()

        # Grad-CAM computation
        pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))
        cam = torch.zeros(self.activations.shape[2:], device=self.device)

        for i, w in enumerate(pooled_gradients):
            cam += w * self.activations[0, i]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(
            cam,
            size=self.input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        cam = cam.squeeze().cpu().detach().numpy()
        return cam, pred_class
