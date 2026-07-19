from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import supervision as sv

from metrics.dice import dice_score
from metrics.iou import iou_score
from metrics.precision_recall import precision_recall
from utils.visualize_predictions import visualize_predictions


model = YOLO(
    "runs/segment/train-3/weights/best.pt"
)

image_dir = Path("/Users/justicealuu/research_project/perovskite_segment/data/solarpark/test/images")
mask_dir = Path("/Users/justicealuu/research_project/perovskite_segment/data/solarpark/test/masks")


dice_total = 0
iou_total = 0
precision_total = 0
recall_total = 0

count = 0

print(image_dir)
print(image_dir.exists())
print(list(image_dir.glob("*"))[:5])


for image_path in sorted(image_dir.glob("*")):

    mask_path = None

    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        candidate = mask_dir / (image_path.stem + ext)
        if candidate.exists():
            mask_path = candidate
            break

    if mask_path is None:
        print(f"Mask not found for {image_path.name}")
        continue

    gt = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    gt = (gt > 127).astype("float32")

    results = model(str(image_path), verbose=False)

    result = results[0]

    if result.masks is None:

        pred = torch.zeros(
            (1, 1, gt.shape[0], gt.shape[1])
        )

    else:

        pred_mask = result.masks.data[0]

        pred_mask = pred_mask.cpu().numpy()

        pred_mask = cv2.resize(
            pred_mask,
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        pred_mask = (pred_mask > 0.5).astype("float32")

        pred = torch.tensor(pred_mask)

        pred = pred.unsqueeze(0).unsqueeze(0)

    gt = torch.tensor(gt)

    gt = gt.unsqueeze(0).unsqueeze(0)

    dice_total += dice_score(pred, gt)
    iou_total += iou_score(pred, gt)
    precision_total += precision_recall(pred, gt)[0]
    recall_total += precision_recall(pred, gt)[1]

    count += 1


print(f"Dice      : {dice_total/count:.4f}")
print(f"IoU       : {iou_total/count:.4f}")
print(f"Precision : {precision_total/count:.4f}")
print(f"Recall    : {recall_total/count:.4f}")

# if torch.cuda.is_available():

#     device = torch.device("cuda")

# elif torch.mps.is_available():

#     device = torch.device("mps")

# else:

#     device = torch.device("cpu")

# visualize_predictions(
#     cfg,
#     model,
#     test_loader,
#     device,   
#     num_images=71
# )

# detections = sv.Detections.from_ultralytics()

# box_annotator = sv.MaskAnnotator()
# labels = [
# 	f"{classes[class_id]} {confidence:0.2f}"
# 	for _, _, confidence, class_id, _
# 	in detections
# ]