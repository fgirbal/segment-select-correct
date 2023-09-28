import random

import numpy as np
import torch

from ssc_ris.select.utils import separate_instance_masks_torch as separate_instance_masks
from .loss_utils import cross_entropy, contrastive_kl_loss, pseudo_intersection_over_union

softmax = torch.nn.Softmax(dim=1)


def random_assign_and_cross_entropy(inputs, outputs, targets, mask_idx_info, params={}):
    assigned_masks = []
    output_loss = torch.Tensor([0.0]).cuda(non_blocking=True)

    for img_idx, (input, output, merged_target) in enumerate(zip(inputs, outputs, targets)):
        all_instance_masks = separate_instance_masks(merged_target)

        same_img_indices = torch.where((inputs.reshape((inputs.shape[0], -1)) == input.reshape(-1)).all(dim=1))[0]
        same_img_same_mask_indices = torch.where(mask_idx_info[same_img_indices] == mask_idx_info[img_idx])[0] + same_img_indices[0].cpu()
        same_img_diff_mask_indices = torch.where(mask_idx_info[same_img_indices] != mask_idx_info[img_idx])[0] + same_img_indices[0].cpu()

        if len(all_instance_masks) == 1:
            # there's only one mask, match it
            output_loss += cross_entropy(output.unsqueeze(0), all_instance_masks[0].unsqueeze(0).to(torch.long))
            assigned_masks.append(0)
        else:
            # there're multiple masks, must assign one of these masks later

            # has this object been assigned a mask already?
            if same_img_same_mask_indices.shape[0] > 0 and same_img_same_mask_indices[0] < img_idx:
                # it has, just use this one and continue
                already_assigned_mask = assigned_masks[same_img_same_mask_indices[0]]
                output_loss += cross_entropy(output.unsqueeze(0), all_instance_masks[already_assigned_mask].unsqueeze(0).to(torch.long))
                assigned_masks.append(already_assigned_mask)
                continue
            
            # no mask has been assigned to this object yet, choose one

            # has a mask for a different object already been assigned? If so, exclude that one from the random draw
            exclude_masks = []
            if same_img_diff_mask_indices.shape[0] > 0 and same_img_diff_mask_indices[0] < img_idx:
                for other_img_idx in same_img_diff_mask_indices:
                    if other_img_idx < img_idx and assigned_masks[other_img_idx] not in exclude_masks:
                        exclude_masks.append(assigned_masks[other_img_idx])

            options = [i for i in range(0, len(all_instance_masks)) if i not in exclude_masks]
            if len(options) == 0:
                print("---- Not enough masks for objects in image :( ----")
                options = [i for i in range(0, len(all_instance_masks))] 

            random_mask_idx = random.choice(options)
            output_loss += cross_entropy(output.unsqueeze(0), all_instance_masks[random_mask_idx].unsqueeze(0).to(torch.long))
            assigned_masks.append(random_mask_idx)

    return output_loss / outputs.shape[0], {}


def greedy_no_constraints_and_cross_entropy(inputs, outputs, targets, mask_idx_info, params={}):
    # no constraints on assigning the same mask to the same output, simply greedy on the output
    assigned_masks = {}
    all_masks = {}
    output_loss = torch.Tensor([0.0]).cuda(non_blocking=True)
    softmaxed_outputs = softmax(outputs)

    for img_idx, (input, output, merged_target) in enumerate(zip(inputs, outputs, targets)):
        all_instance_masks = separate_instance_masks(merged_target)
        all_masks[img_idx] = all_instance_masks

        if len(all_instance_masks) == 1:
            # there's only one mask, match it
            assigned_masks[img_idx] = 0
            output_loss += cross_entropy(output.unsqueeze(0), all_instance_masks[0].unsqueeze(0).to(torch.long))
        else:
            mask_preferences_img = torch.tensor([
                pseudo_intersection_over_union(softmaxed_outputs[img_idx], instance_mask)
                for instance_mask in all_instance_masks
            ])
            assigned_masks[img_idx] = torch.argmax(mask_preferences_img)
            output_loss += cross_entropy(
                output.unsqueeze(0),
                all_instance_masks[assigned_masks[img_idx]].unsqueeze(0).to(torch.long)
            )

    assigned_masks_list = torch.cat([
        all_masks[idx][assigned_masks[idx]].unsqueeze(0) for idx in range(len(assigned_masks)
    )])
    ice_loss = output_loss / outputs.shape[0]
    return ice_loss, {"ice loss": ice_loss, "contrastive loss": torch.tensor(0), "matched masks": assigned_masks_list}


def greedy_match_and_cross_entropy(inputs, outputs, targets, mask_idx_info, params={}):
    assigned_masks = {}
    same_mask_constraints = {}
    diff_mask_constraints = {}
    all_masks = {}
    mask_preferences = {}
    output_loss = torch.Tensor([0.0]).cuda(non_blocking=True)
    softmaxed_outputs = softmax(outputs)

    for img_idx, (input, output, merged_target) in enumerate(zip(inputs, outputs, targets)):
        all_instance_masks = separate_instance_masks(merged_target)

        same_img_indices = torch.where((inputs.reshape((inputs.shape[0], -1)) == input.reshape(-1)).all(dim=1))[0]
        same_img_same_mask_indices = torch.where(mask_idx_info[same_img_indices] == mask_idx_info[img_idx])[0] + same_img_indices[0].cpu()
        same_img_diff_mask_indices = torch.where(mask_idx_info[same_img_indices] != mask_idx_info[img_idx])[0] + same_img_indices[0].cpu()

        if len(all_instance_masks) == 1:
            # there's only one mask, match it
            output_loss += cross_entropy(output.unsqueeze(0), all_instance_masks[0].unsqueeze(0).to(torch.long))
            assigned_masks[img_idx] = 0
        
        same_mask_constraints[img_idx] = same_img_same_mask_indices
        diff_mask_constraints[img_idx] = same_img_diff_mask_indices
        mask_preferences[img_idx] = torch.tensor([
            pseudo_intersection_over_union(softmaxed_outputs[img_idx], instance_mask)
            for instance_mask in all_instance_masks
        ])
        all_masks[img_idx] = all_instance_masks

    # if there are any outputs that have more than one mask, then assign them through Hungarian matching
    for img_idx in same_mask_constraints.keys():
        # this object's mask has been assigned already, no need to do anything
        if img_idx in assigned_masks:
            continue

        # this object's mask has not been choosen yet, assign masks to same and diff mask constraints based on Hungarian matching
        matching_indices = torch.cat((same_mask_constraints[img_idx], diff_mask_constraints[img_idx])).numpy()
        matching_mask_preferences = []
        matching_masks = []
        for idx in matching_indices:
            matching_mask_preferences.append(mask_preferences[idx])
            matching_masks.append(all_masks[idx])

        # matching_mask_preferences = torch.vstack(matching_mask_preferences)
        
        # iteratively select the highest score masks and eliminate the ones that have been selected already
        while True:
            highest_iou_idx = torch.argmax(torch.Tensor([torch.max(idx_masks) for idx_masks in matching_mask_preferences]))
            mask_idx = torch.argmax(matching_mask_preferences[highest_iou_idx])

            # if the best if -Inf, all existing masks have been assigned
            if matching_mask_preferences[highest_iou_idx][mask_idx] == -float('inf'):
                break

            highest_index = matching_indices[highest_iou_idx]
            highest_mask = matching_masks[highest_iou_idx][mask_idx]

            # assign this mask to all same_mask_constraints[highest_index]
            for same_mask_index in same_mask_constraints[highest_index]:
                assigned_masks[int(same_mask_index)] = int(mask_idx)
                output_loss += cross_entropy(outputs[same_mask_index].unsqueeze(0), highest_mask.unsqueeze(0).to(torch.long))

                index = np.where(matching_indices == same_mask_index.numpy())[0][0]
                matching_mask_preferences[index][:] = -float('inf')

            # look for the same mask in diff_mask_constraints and change the IoU to -inf to blacklist this mask
            for diff_mask_index in diff_mask_constraints[highest_index]:
                if diff_mask_index in assigned_masks:
                    continue
                
                diff_mask_matching_index = np.where(matching_indices == diff_mask_index.numpy())[0][0]

                for i, diff_mask in enumerate(matching_masks[diff_mask_matching_index]):
                    if (diff_mask == highest_mask).all():
                        matching_mask_preferences[diff_mask_matching_index][i] = -float('inf')
                        break

    # if there are any masks that have not been assigned, it's because there's not enough masks for the number of objects
    # assign a random mask
    for img_idx in same_mask_constraints.keys():
        # this object's mask has been assigned already, no need to do anything
        if img_idx in assigned_masks:
            continue
    
        random_mask_idx = random.choice(range(len(all_masks[img_idx])))
        assigned_masks[img_idx] = random_mask_idx

    assigned_masks_list = torch.cat([
        all_masks[idx][assigned_masks[idx]].unsqueeze(0) for idx in range(len(assigned_masks)
    )])
    ice_loss = output_loss / outputs.shape[0]
    return ice_loss, {"ice loss": ice_loss, "contrastive loss": torch.tensor(0), "matched masks": assigned_masks_list}


def greedy_match_and_contrastive(inputs, outputs, targets, mask_idx_info, params={"contrastive_alpha": 0.01}):
    ice_loss, ice_log = greedy_match_and_cross_entropy(
        inputs, outputs, targets, mask_idx_info
    )
    matched_masks = ice_log["matched masks"]

    contrastive_loss = contrastive_kl_loss(inputs, outputs, matched_masks, mask_idx_info)

    return ice_loss + params["contrastive_alpha"] * contrastive_loss, {"ice loss": ice_loss, "contrastive loss": contrastive_loss, "matched masks": matched_masks}
