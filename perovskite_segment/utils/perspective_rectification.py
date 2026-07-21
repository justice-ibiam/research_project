import cv2
import numpy as np

def find_panel_contour(mask):
    """
    mask: HxW binary image (0 or 255)
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    return max(contours, key=cv2.contourArea)

def get_panel_corners(contour):

    hull = cv2.convexHull(contour)

    epsilon = 0.01 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(
        hull,
        epsilon,
        True
    )

    print(len(approx))
    print(approx.reshape(-1,2))



    # while epsilon <= 0.05 * cv2.arcLength(contour, True):

    #     approx = cv2.approxPolyDP(
    #         contour,
    #         epsilon,
    #         True
    #     )

    #     if len(approx) == 4:
    #         corners = approx.reshape(4, 2).astype(np.float32)
    #         return order_points(corners)

    #     epsilon += 0.005 * cv2.arcLength(contour, True)

    # Fallback if approximation never reaches 4 points
    rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect)
    corners = order_points(corners.astype(np.float32))

    return corners

def order_points(pts):

    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)

    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4,2), dtype=np.float32)

    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left

    return ordered

def rectify_image(image,
                  corners,
                #   width=320,
                #   height=240
                  ):

    # corners = order_points(corners)
    print(image.shape)
    print("Corners inside rectify_image:")
    print(corners)

    widthA = np.linalg.norm(corners[2] - corners[3])
    widthB = np.linalg.norm(corners[1] - corners[0])
    width = int(max(widthA, widthB))

    heightA = np.linalg.norm(corners[1] - corners[2])
    heightB = np.linalg.norm(corners[0] - corners[3])
    height = int(max(heightA, heightB))
    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(
        corners,
        destination
    )

    rectified = cv2.warpPerspective(
        image,
        M,
        (width,height)
    )

    return rectified

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