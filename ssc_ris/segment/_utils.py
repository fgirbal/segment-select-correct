from PIL import Image
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def save_all_masks(
            image_path: str,
            binary_masks: np.ndarray,
            sentences: List[str],
            output_name: str = "mask_example.png"
            ) -> None:
    """Save all masks in a Matplotlib viz for debugging purposes.

    Args:
        image_path (str): input image location on file
        binary_masks (np.ndarray): array with all the binary masks
        sentences (List[str]): list of sentences that generated the binary masks
        output_name (str, optional): name of the outputted figure. Defaults to "mask_example.png".
    """
    # 1 subplot for the image itself, 1 for the overall segmentation mask, and 1 per binary segmentation masks
    fig, axs = plt.subplots(1, 1+len(binary_masks), figsize=(4*(1+len(binary_masks)), 4))

    img = Image.open(image_path)
    for ax in axs:
        ax.imshow(img)
        ax.axis('off')

    for i, mask in enumerate(binary_masks):
        axs[1+i].imshow(mask, alpha=0.4)
        axs[1+i].set_title(sentences[i])

    plt.tight_layout()
    plt.savefig(output_name)

