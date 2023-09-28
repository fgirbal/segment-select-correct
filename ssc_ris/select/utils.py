from typing import List

import torch
import numpy as np


def separate_instance_masks(instance_merged_mask: np.ndarray) -> List[np.ndarray]:
    """Given an image with all instance masks, split them into separate binary masks.

    Args:
        instance_merged_mask (np.ndarray): input n-dim mask

    Returns:
        List[np.ndarray]: binary mask
    """
    instance_vals = np.unique(instance_merged_mask)
    return [
        (instance_merged_mask == val).astype(np.int32)
        for val in instance_vals if val != 0.0
    ]

def separate_instance_masks_torch(instance_merged_mask: torch.Tensor) -> List[torch.Tensor]:
    """Given an image with all instance masks, split them into separate binary masks.

    Args:
        instance_merged_mask (torch.Tensor): input n-dim mask

    Returns:
        List[torch.Tensor]: binary mask
    """
    instance_vals = torch.unique(instance_merged_mask)
    return [
        (instance_merged_mask == val).to(torch.int32)
        for val in instance_vals if val != 0.0
    ]
