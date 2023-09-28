import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from .dataset import ReferDataset
from .unsupervised_dataset import UnsupervisedReferDataset
import ssc_ris.utils.transforms as T

# --- Fetching functions ---

def get_dataset(
            dataset: str,
            dataset_root: str,
            data_split: str,
            transforms: object = None,
            split_by: str = 'unc',
            return_attributes: bool = True,
            eval_model: bool = False
            ) -> Tuple[ReferDataset, int]:
    """Obtains a ReferDataset object

    Args:
        dataset (str): dataset descriptior
        dataset_root (str): root for the dataset
        data_split (str): dataset split
        transforms (object, optional): transforms to apply to image data. Defaults to None.
        split_by (str, optional): RIS dataset splits ('unc', 'umd', 'google'). Defaults to 'unc'.
        return_attributes (bool, optional): if True, extra dataset attributes are returned. Defaults to True.
        eval_model (bool, optional): if True, all referring sentences are returned. Defaults to False.

    Returns:
        Tuple[ReferDataset, int]: dataset object and number of classes
    """
    ds = ReferDataset(
        dataset,
        refer_data_root=dataset_root,
        splitBy=split_by,
        split=data_split,
        image_transforms=transforms,
        target_transforms=None,
        return_attributes=return_attributes,
        eval_mode=eval_model
    )
    num_classes = 2

    return ds, num_classes

def get_unsupervised_dataset(
            dataset: str,
            dataset_root: str,
            pseudo_masks_root: str,
            data_split: str,
            transforms: object = None,
            split_by: str = 'unc',
            one_sentence: bool = True
            ) -> Tuple[UnsupervisedReferDataset, int]:
    ds = UnsupervisedReferDataset(
        dataset,
        refer_data_root=dataset_root,
        pseudo_masks_data_root=pseudo_masks_root,
        splitBy=split_by,
        split=data_split,
        image_transforms=transforms,
        target_transforms=None,
        one_sentence=one_sentence
    )
    num_classes = 2

    return ds, num_classes


def get_transform(img_size: int) -> object:
    """Get the dataset transform.

    Args:
        img_size (int): size of the image

    Returns:
        object: composed transform
    """
    transforms = [
        T.Resize(img_size, img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    return T.Compose(transforms)

# --- Saving unsupervised masks functions ---

def save_binary_object_masks(object_masks: List[np.ndarray], ref_ids: List[str], output_mask_folder: str) -> None:
    """Save the list of binary masks.

    Args:
        object_masks (List[np.ndarray]): list of masks to save
        ref_ids (List[str]): ids of the masks to save
        output_mask_folder (str): folder where to save the masks
    """
    for mask, ref_id in zip(object_masks, ref_ids):
        save_binary_object_mask(mask, ref_id, output_mask_folder)

def save_binary_object_mask(mask: np.ndarray, ref_id: str, output_mask_folder: str) -> None:
    """Save the list of binary masks.

    Args:
        object_masks (List[np.ndarray]): list of masks to save
        ref_ids (List[str]): ids of the masks to save
        output_mask_folder (str): folder where to save the masks
    """
    img_name = os.path.join(output_mask_folder, f"pseudo_gt_mask_{ref_id}.png")
    plt.imsave(img_name, mask, cmap=cm.gray)
