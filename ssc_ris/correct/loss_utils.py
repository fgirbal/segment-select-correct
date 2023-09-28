import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

kl_loss = nn.KLDivLoss(reduction="batchmean")
cos_sim = torch.nn.CosineSimilarity(dim=0)


def cross_entropy(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target, weight=weight)


def pseudo_intersection_over_union(prediction, target_mask):
    # from Recurrent Instance Segmentation paper (https://arxiv.org/pdf/1511.08250.pdf)
    # assumes prediction is the result of a softmax function
    y_hat_y_dot = (prediction[1] * target_mask).sum()
    return y_hat_y_dot / torch.max(prediction[1].sum() + target_mask.sum() - y_hat_y_dot, torch.tensor(1e-7).to(y_hat_y_dot.device))


def contrastive_kl_loss(batch_images, output, batch_targets, batch_mask_idx_info):
    # compute KL divergence on negative terms; don't add anything for positive terms
    negative_examples_constant = 0.1
    contrastive_loss = torch.Tensor([0.0]).cuda(non_blocking=True)
    output = torch.softmax(output, dim=1)

    n_terms = 0
    for img_idx, img in enumerate(batch_images):
        same_img_indices = torch.where((batch_images.reshape((batch_images.shape[0], -1)) == img.reshape(-1)).all(dim=1))[0]

        same_img_same_mask_indices = torch.where(batch_mask_idx_info[same_img_indices] == batch_mask_idx_info[img_idx])[0] + same_img_indices[0].cpu()
        same_img_diff_mask_indices = torch.where(batch_mask_idx_info[same_img_indices] != batch_mask_idx_info[img_idx])[0] + same_img_indices[0].cpu()

        # sum masks of the same object
        for other_pos_img_idx in same_img_same_mask_indices:
            if img_idx == other_pos_img_idx:
                continue

            contrastive_loss += kl_loss(torch.log(output[img_idx]), output[other_pos_img_idx]) / (output[img_idx].shape[1] * output[img_idx].shape[2])
            n_terms += 1

        # 1 / kl for masks of different objects
        for other_neg_img_idx in same_img_diff_mask_indices:
            active_pixels = (batch_targets[img_idx] == 1) | (batch_targets[other_neg_img_idx] == 1)
            masked_img_output = output[img_idx][:, active_pixels]
            masked_other_img_output = output[other_neg_img_idx][:, active_pixels]

            # hinge loss combined with an inverse KL to incentivize low similarity
            negative_loss_term = torch.min(
                torch.Tensor([1]).to(masked_img_output.device),
                1 / (negative_examples_constant * kl_loss(torch.log(masked_img_output), masked_other_img_output))
            )
            contrastive_loss += negative_loss_term

            n_terms += 1
        
        # save_batch_images_and_masks(batch_images.cpu(), batch_targets.cpu(), 'example.png')

    if n_terms:
        contrastive_loss /= n_terms

    return contrastive_loss

