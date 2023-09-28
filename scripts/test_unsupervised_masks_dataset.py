# Testing script for Stages 1 and 2
import os
import json
import argparse
import pickle
from PIL import Image
import random

from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import nltk
import clip
import cv2

from ssc_ris.refer_dataset.utils import get_dataset
from ssc_ris.select.utils import separate_instance_masks
from ssc_ris.select import vp_and_max_sim_clip_choice


colormap = np.array([
    [  0,   0,   0, 0],
    [ 245, 233, 66, 128]
], dtype=np.float32)
colormap /= 255.0
seg_colormap = ListedColormap(colormap)

def plot_img_pseudo_gt(image, pseudo_mask, gt_mask, output_filename):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(image)
    axs[0].imshow(gt_mask, cmap=seg_colormap)
    axs[0].set_title('GT')
    axs[0].axis('off')

    axs[1].imshow(image)
    axs[1].imshow(pseudo_mask, cmap=seg_colormap)
    axs[1].set_title('Pseudo')
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_filename)

    fig.clf()
    plt.close()


def best_mask_choice(masks, gt_mask):
    print("------------- WARNING: best mask being chosen w.r.t. GT, should only be used for testing -------------")

    best_mask_iou = -np.Inf
    best_mask = None
    for pseudo_mask_ in masks:
        intersect = (gt_mask & pseudo_mask_)
        intersect_area = intersect.sum().astype(np.float32)
        union = (gt_mask | pseudo_mask_)

        iou = intersect_area / union.sum()
        if iou > best_mask_iou:
            best_mask_iou = iou
            best_mask = pseudo_mask_

    return best_mask


def random_mask_choice(image, masks):
    return random.choice(masks)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-s', '--stage', type=str, required=True, choices=["segment", "select"])
    parser.add_argument('-d', '--dataset', type=str, default="refcoco")
    parser.add_argument('--dataset-root', type=str, default="refer/data")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--split-by', type=str, default="unc", choices=["unc", "umd", "google"])

    parser.add_argument(
        '--mask-choice',
        type=str,
        default="best",
        choices=[
            "best",
            "random",
            "red_ellipse",
            "red_ellipse_vit_32",
            "rectangle",
            "red_dense_mask",
            "reverse_blur",
            "reverse_blur_vit_32"
        ]
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    experiment_name = args.name
    dataset = args.dataset
    dataset_root = args.dataset_root
    data_split = args.split
    img_size = 480
    
    if args.stage == "segment":
        trail = "segment_stage_masks"
    else:
        trail = "select_stage_masks"

    output_path = f"{trail}/{experiment_name}"
    if data_split != "train":
        output_path = f"{trail}_{data_split}/{experiment_name}"
    
    pseudo_mask_folder = f"{output_path}/"
    if dataset != "refcocog":
        pseudo_mask_folder += f"instance_masks/{dataset}"
    else:
        pseudo_mask_folder += f"instance_masks/{dataset}_{args.split_by}"

    old_dataset, num_classes = get_dataset(
        dataset,
        dataset_root,
        data_split,
        split_by=args.split_by
    )
    coco_images_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), old_dataset.refer.IMAGE_DIR)

    n_masks_per_image = []
    empty_masks = 0
    intersection = []
    iou = []
    cum_I, cum_U = 0, 0

    if data_split == "train":
        range_end = 1000
    else:
        range_end = len(old_dataset)

    if args.mask_choice not in ["best", "random"]:
        print("loading CLIP...")
        device = "cuda:0"
        if args.mask_choice == "reverse_blur_vit_32" or args.mask_choice == "red_ellipse_vit_32":
            clip_model, preprocess = clip.load("ViT-B/32", device=device)
        else:
            clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)
        
        print("loaded")

    for i in tqdm(range(range_end)):
        dataset_object = old_dataset[i]

        ref_id = dataset_object[-1]['ref_id']
        gt_mask = np.array(dataset_object[1])

        pseudo_mask_img_name = os.path.join(pseudo_mask_folder, f"pseudo_gt_mask_{ref_id}.png")
        instance_merged_pseudo_mask = np.array(Image.open(pseudo_mask_img_name).convert('L'))

        all_instance_masks = separate_instance_masks(instance_merged_pseudo_mask)
        n_masks_per_image.append(len(all_instance_masks))

        if len(all_instance_masks) == 0:
            intersection.append(0.0)
            iou.append(0.0)
        else:
            if args.mask_choice == "best":
                chosen_mask = best_mask_choice(all_instance_masks, gt_mask)
            elif args.mask_choice == "random":
                chosen_mask = random_mask_choice(dataset_object[0], all_instance_masks)
            else:
                chosen_mask = vp_and_max_sim_clip_choice(
                    clip_model,
                    preprocess,
                    dataset_object[0],
                    all_instance_masks,
                    dataset_object[-1]['sentences_sent'],
                    method=args.mask_choice
                )

                # couldn't decide on a mask, return an empty one
                if chosen_mask is None:
                    chosen_mask = np.zeros_like(instance_merged_pseudo_mask)

            intersect = (gt_mask & chosen_mask)
            intersect_area = intersect.sum().astype(np.float32)
            union = (gt_mask | chosen_mask)
            union_area = union.sum()

            intersection.append(intersect_area / gt_mask.sum())
            iou.append(intersect_area / union_area)
            cum_I += intersect_area
            cum_U += union_area

        if instance_merged_pseudo_mask.sum() == 0:
            empty_masks += 1

    print('mIoU:', np.array(iou).mean())
    print('oIoU:', cum_I / cum_U)

    n_masks_per_image = np.array(n_masks_per_image)
    print('------')
    print('mean # masks:', n_masks_per_image.mean())
    print('max # masks:', n_masks_per_image.max())

    data_stats = {
        'mIoU': np.array(iou).mean(),
        'oIoU': cum_I / cum_U,
        'mean # masks': n_masks_per_image.mean(),
        'max # masks': float(n_masks_per_image.max())
    }

    with open(output_path + f"/test_results_{args.mask_choice}_{dataset}_{args.split_by}.json", 'w') as f:
        json.dump(data_stats, f, indent=4)
