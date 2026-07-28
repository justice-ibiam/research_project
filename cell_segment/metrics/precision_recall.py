import torch


def precision_recall(pred, target):

    pred = (torch.sigmoid(pred) > 0.5).float()

    tp = (
        (pred == 1)
        &
        (target == 1)
    ).sum()

    fp = (
        (pred == 1)
        &
        (target == 0)
    ).sum()

    fn = (
        (pred == 0)
        &
        (target == 1)
    ).sum()

    precision = tp / (tp + fp + 1e-6)

    recall = tp / (tp + fn + 1e-6)

    return (
        precision.item(),
        recall.item()
    )