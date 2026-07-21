import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np

def remove_background(image, mask,
                      kernel_size=5,
                      dilation_iters=1):
    """
    Remove the background using a predicted segmentation mask.

    Parameters

    image : np.ndarray
        Original grayscale or RGB image.

    mask : np.ndarray
        Predicted binary mask (0/255 or probabilities).

    kernel_size : int
        Size of dilation kernel.

    dilation_iters : int
        Number of dilation iterations.

    Returns

    foreground : np.ndarray
        Image with the background removed.

    clean_mask : np.ndarray
        Processed binary mask.
    """


    # Convert to binary mask
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)


    # Keep only largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean_mask = np.zeros_like(mask)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_mask[labels == largest] = 255


    # Dilate slightly to preserve borders
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    clean_mask = cv2.dilate(
        clean_mask,
        kernel,
        iterations=dilation_iters
    )


    # Apply mask
    if image.ndim == 2:
        foreground = cv2.bitwise_and(image, image, mask=clean_mask)
    else:
        foreground = cv2.bitwise_and(image, image, mask=clean_mask)

    return foreground, clean_mask

def letterbox(img, size=512):
    h,w = img.shape[:2]
    s = min(size/w, size/h); nw,nh = int(w*s), int(h*s)
    rsz = cv2.resize(img,(nw,nh))
    canvas = np.zeros((size,size,3),np.uint8)
    top,left = (size-nh)//2,(size-nw)//2
    canvas[top:top+nh, left:left+nw] = rsz
    return canvas,(top,left,nh,nw),(h,w)

def _order_pts(pts):
    pts = np.asarray(pts,np.float32)
    s=pts.sum(1); d=np.diff(pts,1).ravel()
    tl,br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr,bl = pts[np.argmin(d)], pts[np.argmax(d)]
    return np.array([tl,tr,br,bl],np.float32)

def _largest_cnt(mask):
    m = (mask>0).astype(np.uint8)
    k = np.ones((7,7),np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts,key=cv2.contourArea) if cnts else None

def _quad_from_contour(cnt):
    hull = cv2.convexHull(cnt); peri = cv2.arcLength(hull, True)
    for frac in np.linspace(0.01,0.08,8):
        approx = cv2.approxPolyDP(hull, epsilon=frac*peri, closed=True)
        if len(approx)==4:
            return approx.reshape(-1,2).astype(np.float32)
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype(np.float32)

def cutout_rgba(img, mask):
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:,:,3] = mask
    return rgba

def rectify_image(foreground, mask):
    pad=20
    out_width = 160
    rows, cols = (4,5)
    cnt = _largest_cnt(mask)
    if cnt is None or cv2.contourArea(cnt)<200:
        return None, None, None
    src = _order_pts(_quad_from_contour(cnt))
    W   = int(out_width + 2*pad)
    H   = int((rows/cols)*out_width + 2*pad)
    dst = np.array([[pad,pad],[W-pad-1,pad],[W-pad-1,H-pad-1],[pad,H-pad-1]], np.float32)
    M   = cv2.getPerspectiveTransform(src, dst)
    rect_bgr  = cv2.warpPerspective(foreground, M, (W,H), flags=cv2.INTER_LINEAR)
    rect_mask = cv2.warpPerspective(mask,   M, (W,H), flags=cv2.INTER_NEAREST)
    rect_rgba = cutout_rgba(rect_bgr, (rect_mask>0).astype(np.uint8)*255)
    rect_grid = rect_bgr.copy()
    for c in range(1, cols):
        x = int(pad + c*(W-2*pad)/cols); cv2.line(rect_grid, (x,pad),(x,H-pad-1),(0,0,0),2)
    for r in range(1, rows):
        y = int(pad + r*(H-2*pad)/rows); cv2.line(rect_grid, (pad,y),(W-pad-1,y),(0,0,0),2)
    cv2.rectangle(rect_grid,(pad,pad),(W-pad-1,H-pad-1),(0,0,0),3)
    rect_bgr  = rect_bgr[int(0.02*H):, :]
    rect_rgba = rect_rgba[int(0.02*H):, :]
    rect_grid = rect_grid[int(0.02*H):, :]

    return rect_rgba

def visualize_rectified( cfg, image, pred, gt, rectified, num_images, count):

    fig, ax = plt.subplots(
        1,
        4,
        figsize=(16,4)
    )


    image = image * np.array(cfg.dataset.std) + np.array(cfg.dataset.mean)
    image = np.clip(image, 0, 1)
    ax[0].imshow(image)
    ax[0].set_title("Thermal Image")
    ax[0].axis("off")

    ax[1].imshow(gt, cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title("Prediction")
    ax[2].axis("off")


    rgb = rectified[:, :, :3]
    alpha = rectified[:, :, 3:] / 255.0

    rgb = rgb * np.array(cfg.dataset.std) + np.array(cfg.dataset.mean)
    rgb = np.clip(rgb, 0, 1)

    rectified = np.concatenate([rgb, alpha], axis=2)
    ax[3].imshow(rectified, cmap="gray")
    ax[3].set_title("Rectified")
    ax[3].axis("off")

    plt.tight_layout()
    plt.show()

    count += 1

    if count >= num_images:
        return