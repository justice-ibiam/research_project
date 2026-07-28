import torch

def ordinal_targets(y, num_classes=5):
    """
    y: (B,) labels in {0,1,2,3,4}
    returns: (B, num_classes-1)
    """
    B = y.size(0)
    targets = torch.zeros(B, num_classes - 1, device=y.device)

    for k in range(1, num_classes):
        targets[:, k - 1] = (y >= k).float()

    return targets

def compute_pos_weight(loader, num_classes, device):
    counts = torch.zeros(num_classes - 1)
    total = 0

    for _, y in loader:
        y = y.long()
        for k in range(num_classes - 1):
            counts[k] += (y > k).sum()
        total += y.size(0)

    pos_weight = (total - counts) / (counts + 1e-6)
    return pos_weight.to(device)
