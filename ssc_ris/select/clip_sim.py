from PIL import Image
from typing import List, Tuple

import numpy as np
import torch
import clip

from .visual_prompting_fns import vp_rectangle, vp_red_ellipse, vp_dense_mask, vp_reverse_blur


def vp_and_max_sim_clip_choice(
            clip_model: torch.nn.Module,
            preprocess: torch.nn.Module,
            image: Image,
            masks: List[np.ndarray],
            sentences: List[str],
            method: str = "reverse_blur",
            thickness: int = 4,
            margin: int = 4,
            alpha: float = 0.25,
            color: Tuple[int, int, int] = (255, 0, 0),
            std_dev: int = 50
    ) -> np.ndarray:
    """Visually prompt and select the mask with the highest CLIP similarity.

    Args:
        clip_model (torch.nn.Module): CLIP model
        preprocess (torch.nn.Module): CLIP pre-processing module
        image (Image): image
        masks (List[np.ndarray]): list of candidate masks
        sentences (List[str]): sentences to encode and compute similarity to
        method (str, optional): visual prompting method. Defaults to "reverse_blur".
        thickness (int, optional): "rectangle"/"red_ellipse" prompting parameter. Defaults to 4.
        margin (int, optional): "rectangle"/"red_ellipse" prompting parameter. Defaults to 4.
        alpha (float, optional): "red_dense_mask" prompting parameter. Defaults to 0.25.
        color (Tuple[int, int, int], optional): "red_dense_mask" prompting parameter. Defaults to (255, 0, 0).
        std_dev (int, optional): "reverse_blur" prompting parameter. Defaults to 50.

    Returns:
        np.ndarray: the highest similarity mask
    """
    if len(masks) == 1:
        return masks[0]
    
    device = next(clip_model.parameters()).device
    with torch.no_grad():
        sentences_embedding = clip_model.encode_text(clip.tokenize(sentences).to(device))

    sentences_embedding /= sentences_embedding.norm(dim=-1, keepdim=True)

    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    best_similarity = -np.Inf
    best_mask = None
    for mask in masks:
        if mask.sum() < 500:
            continue

        if method == "rectangle":
            annotated_img = vp_rectangle(
                open_cv_image,
                mask,
                thickness=thickness,
                margin=margin
            )
        elif method == "red_ellipse":
            annotated_img = vp_red_ellipse(
                open_cv_image,
                mask,
                thickness=thickness,
                margin=margin
            )
        elif method == "red_dense_mask":
            annotated_img = vp_dense_mask(
                open_cv_image,
                mask,
                alpha=alpha,
                color=color
            )
        elif method == "reverse_blur":
            annotated_img = vp_reverse_blur(
                open_cv_image,
                mask,
                std_dev=std_dev
            )
        else:
            raise NotImplemented

        # get the CLIP image embedding
        image_input = preprocess(Image.fromarray(annotated_img)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ sentences_embedding.T).mean()

        if similarity >= best_similarity:
            best_mask = mask
            best_similarity = similarity

    return best_mask
