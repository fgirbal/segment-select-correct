from typing import Tuple

import cv2
import numpy as np


def vp_rectangle(original_image: np.ndarray, mask: np.ndarray, thickness: int = 4, margin: int = 4) -> np.ndarray:
    """Visual prompting using a red rectangle

    Args:
        original_image (np.ndarray): image
        mask (np.ndarray): dense segmentation mask
        thickness (int, optional): thickness of the rectangle. Defaults to 4.
        margin (int, optional): margin of the rectangle w.r.t. the bounding box of the mask. Defaults to 4.

    Returns:
        np.ndarray: annotated image
    """
    mask_indices = np.where(mask == 1)
    start_corner = (mask_indices[1].min()-margin, mask_indices[0].min()-margin)
    end_corner = (mask_indices[1].max()+margin, mask_indices[0].max()+margin)

    annotated_img = cv2.rectangle(original_image.copy(), start_corner, end_corner, (0, 0, 255), thickness)
    return annotated_img

def vp_red_ellipse(original_image: np.ndarray, mask: np.ndarray, thickness: int = 4, margin: int = 4) -> np.ndarray:
    """Visual prompting using a red ellipse

    Args:
        original_image (np.ndarray): image
        mask (np.ndarray): dense segmentation mask
        thickness (int, optional): thickness of the ellipse. Defaults to 4.
        margin (int, optional): margin of the ellipse w.r.t. the bounding box of the mask. Defaults to 4.

    Returns:
        np.ndarray: annotated image
    """
    mask_indices = np.where(mask == 1)
    start_corner = (mask_indices[1].min()-margin, mask_indices[0].min()-margin)
    end_corner = (mask_indices[1].max()+margin, mask_indices[0].max()+margin)

    annotated_img = cv2.ellipse(
        original_image.copy(),
        np.array(((start_corner[0] + end_corner[0]) / 2, (start_corner[1] + end_corner[1]) / 2), dtype=np.int64),
        np.array(((end_corner[0] - start_corner[0]) / 2 + margin, (end_corner[1] - start_corner[1]) / 2 + margin), dtype=np.int64),
        0,
        0,
        360,
        (0, 0, 255),
        thickness
    )
    return annotated_img

def vp_dense_mask(original_image: np.ndarray, mask: np.ndarray, alpha: float = 0.25, color: Tuple[float, float, float] = (255, 0, 0)) -> np.ndarray:
    """Visual prompting using a dense mask

    Args:
        original_image (np.ndarray): image
        mask (np.ndarray): dense segmentation mask
        alpha (float, optional): transparency of the mask. Defaults to 0.25.
        color (Tuple[float, float, float], optional): color of the mask. Defaults to (255, 0, 0).

    Returns:
        np.ndarray: annotated image
    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)

    masked = np.ma.MaskedArray(original_image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    annotated_img = cv2.addWeighted(original_image, 1 - alpha, image_overlay, alpha, 0)

    return annotated_img

def vp_reverse_blur(original_image: np.ndarray, mask: np.ndarray, std_dev: int = 100) -> np.ndarray:
    """Visual prompting using the reverse blur mechanism from the Fine-Tuning paper

    Args:
        original_image (np.ndarray): image
        mask (np.ndarray): dense segmentation mask
        std_dev (int, optional): standard deviation of the Gaussian kernel used for noise. Defaults to 100.

    Returns:
        np.ndarray: annotated image
    """
    blur_background = cv2.GaussianBlur(original_image.copy(), [0, 0], sigmaX=std_dev, sigmaY=std_dev)

    masked_blur = cv2.bitwise_and(blur_background, blur_background, mask=(255*(1-mask)).astype(np.uint8))
    masked_object = cv2.bitwise_and(original_image, original_image, mask=(255*mask).astype(np.uint8))
    annotated_img = masked_blur + masked_object

    return annotated_img