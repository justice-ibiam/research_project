import cv2
import numpy as np


def ben_graham_preprocess(image, sigma=10, scale=None):
    """
    Apply Benjamin Graham preprocessing to fundus images.
    
    Parameters:
    image : numpy.ndarray
        Input fundus image (can be BGR, RGB, or grayscale)
    sigma : int
        Gaussian filter sigma for local color averaging (default: 10)
    scale : int or None
        Output size (scale x scale). If None, keeps original size
        
    Returns:
    numpy.ndarray
        Preprocessed image (uint8)
    """
    # Handle PIL Image if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Ensure 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    
    # Convert to float32
    image = image.astype(np.float32)
    
    # Calculate Gaussian kernel size
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Subtract local average color
    local_avg = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    image = image - local_avg
    
    # Clip extreme values
    image = np.clip(image, -255, 255)
    
    # Normalize to 0-255 range
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    image = (image * 255).astype(np.uint8)
    
    # Resize if scale specified
    if scale is not None:
        image = cv2.resize(image, (scale, scale), interpolation=cv2.INTER_CUBIC)
    
    return image


def crop_fundus_circle(image, threshold=10):
    """
    Crop fundus image to focus on the circular region.
    
    Parameters:
    image : numpy.ndarray
        Input fundus image
    threshold : int
        Threshold for detecting non-black pixels (default: 10)
        
    Returns:
    numpy.ndarray
        Cropped image centered on fundus region
    """
    # Handle PIL Image if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Convert to grayscale for masking
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create mask of non-black regions
    mask = gray > threshold
    
    # Find coordinates of all non-zero pixels
    coords = np.column_stack(np.where(mask))
    
    if len(coords) == 0:
        return image
    
    # Find bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Calculate center and radius
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2
    radius = min(center_y - y_min, y_max - center_y, 
                 center_x - x_min, x_max - center_x)
    
    # Crop to square around the circle
    y1 = max(0, center_y - radius)
    y2 = min(image.shape[0], center_y + radius)
    x1 = max(0, center_x - radius)
    x2 = min(image.shape[1], center_x + radius)
    
    cropped = image[y1:y2, x1:x2]
    
    return cropped

def crop_fundus_circle_hough(image, dp=1.1, min_dist=200, 
                             param1=50, param2=30, 
                             min_radius=100, max_radius=0):
    """
    Robustly crop fundus image using Hough Circle detection.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Hough Circle Transform
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    # If no circle detected, return original image
    if circles is None:
        print(" No circle detected — returning original image.")
        return image

    circles = np.round(circles[0, :]).astype("int")
    x, y, r = circles[0]   # select the first detected circle

    # Crop to square around the circle
    y1 = max(0, y - r)
    y2 = min(image.shape[0], y + r)
    x1 = max(0, x - r)
    x2 = min(image.shape[1], x + r)

    cropped = image[y1:y2, x1:x2]

    return cropped