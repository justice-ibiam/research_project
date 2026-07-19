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
                  width=320,
                  height=240):

    # corners = order_points(corners)
    print(image.shape)
    print("Corners inside rectify_image:")
    print(corners)
    destination = np.array([
        [0,0],
        [width-1,0],
        [width-1,height-1],
        [0,height-1]
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

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,5))

    plt.subplot(121)
    plt.imshow(image)
    plt.title("Original")

    plt.subplot(122)
    plt.imshow(rectified)
    plt.title("Rectified")

    plt.show()