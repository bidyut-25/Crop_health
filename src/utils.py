import cv2
import numpy as np
from skimage.measure import label, regionprops

def preprocess_image(image_path, size=(128, 128)):
    """
    Load and preprocess an image for model inference.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    return img_rgb,img_normalized

def overlay_mask(image, binary_mask):
    """
    Overlay red mask and bounding boxes on the original image.
    """
    if binary_mask.ndim == 3:
        binary_mask = binary_mask.squeeze()

    # Red mask where binary_mask == 1
    red_mask = np.zeros_like(image)
    red_mask[..., 0] = 255  # Red channel full

    # Blend red mask using np.where
    masked_overlay = np.where(binary_mask[..., None] > 0.5, red_mask, image)

    # Bounding boxes
    labeled_mask = label(binary_mask)
    boxed_image = masked_overlay.copy()

    for region in regionprops(labeled_mask):
        if region.area >= 50:
            minr, minc, maxr, maxc = region.bbox
            cv2.rectangle(boxed_image, (minc, minr), (maxc, maxr), (255, 0, 0), 2)

    return boxed_image

