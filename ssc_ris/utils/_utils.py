import re
from typing import Tuple

import torch
import numpy as np
import wandb

from ssc_ris.refer_dataset.bert.modeling_bert import BertModel


def IoU(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Compute IoU.

    Args:
        pred (torch.Tensor): a 2 class segmentation model output
        gt (torch.Tensor): the groundtruth w.r.t. which we'll compute the IoU

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor]: iou, intersection and union
    """
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


def get_sentence_from_tokens(token_sequence: torch.Tensor, bert_model: BertModel) -> str:
    """Given a token sequence encoded using bert_model, decode it as a string.

    Args:
        token_sequence (torch.Tensor): input token sequence
        bert_model (BertModel): Bert model used to encode it; must include a tokenizer.

    Returns:
        str: decoded sequence, stripped of special tokens
    """
    actual_sentence = bert_model.tokenizer.decode(token_sequence)
    actual_sentence = actual_sentence.replace('[CLS]', '').replace('[PAD]', '').replace('[SEP]', '')[1:]
    actual_sentence = re.sub(" +", " ", actual_sentence)
    return actual_sentence


def wandb_mask(
            image: np.ndarray,
            sentence: torch.Tensor,
            pred_mask: np.ndarray,
            true_mask: np.ndarray,
            matched_mask: np.ndarray,
            bert_model: BertModel
            ) -> wandb.Image:
    """Obtain a wandb image which includes the predicted mask, true mask as well as mask options.

    Args:
        image (np.ndarray): original image
        sentence (torch.Tensor): token sequence describing the original sentence
        pred_mask (np.ndarray): predicted mask
        true_mask (np.ndarray): ground-truth/pseudo-ground-truth mask
        matched_mask (np.ndarray): matched mask
        bert_model (BertModel): Bert model used to encode sentence

    Returns:
        wandb.Image: output wandb image for logging
    """
    actual_sentence = get_sentence_from_tokens(sentence, bert_model)

    if matched_mask is None:
        return wandb.Image(
            image,
            masks={
                "prediction" : {"mask_data" : pred_mask, "class_labels" : {1: "object"}},
                "pseudo GT" : {"mask_data" : true_mask + 1, "class_labels" : {obj_int + 1: f"instance {obj_int}" for obj_int in range(1, 10)}}
            },
            caption=actual_sentence
        )
    else:
        # masks must be int64, can't be int32
        matched_mask = matched_mask.astype(np.int64)
        
        return wandb.Image(
            image,
            masks={
                "prediction" : {"mask_data" : pred_mask, "class_labels" : {1: "pred object"}},
                "matched mask": {"mask_data": matched_mask + 1, "class_labels": {2: "ice_object"}},
                "mask options": {"mask_data" : true_mask + 2, "class_labels" : {obj_int + 2: f"instance {obj_int}" for obj_int in range(1, 10)}}
            },
            caption=actual_sentence
        )