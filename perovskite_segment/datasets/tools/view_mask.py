import cv2
import matplotlib.pyplot as plt
mask_path = "/Users/justicealuu/research_project/perovskite_segment/data/cell_segmentation/test/masks/00030_png.rf.a83af340b6827d8250c8eb24811d163d.png"
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

plt.figure(figsize=(8,6))
plt.imshow(mask, cmap="tab20")
plt.colorbar()
plt.show()