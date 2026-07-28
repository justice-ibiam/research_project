import torch
import torch.nn.functional as F

class BinaryComplementCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """
        logits: raw model outputs (B,)
        targets: 0 or 1 labels (B,)
        """
        # Complement target: flip 0 <-> 1
        comp_targets = 1 - targets

        # Convert to float
        comp_targets = comp_targets.float()

        # Standard BCE but using complement target
        loss = F.binary_cross_entropy_with_logits(logits, comp_targets)

        return loss