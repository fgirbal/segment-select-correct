# Select: Stage 2 of the three-stage framework presented
import os
import argparse
from PIL import Image
import random

from tqdm import tqdm
import numpy as np
import clip

from ssc_ris.refer_dataset.utils import get_dataset, save_binary_object_mask
from ssc_ris.select.utils import separate_instance_masks
from ssc_ris.select import vp_and_max_sim_clip_choice

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


def first_mask_choice(image, masks):
    return masks[0]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--original-name', type=str, required=True)
    parser.add_argument('--new-name', type=str, required=True)
    parser.add_argument('-f', '--full', action='store_true')

    parser.add_argument('-d', '--dataset', type=str, default="refcoco")
    parser.add_argument('--dataset-root', type=str, default="refer/data")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--split-by', type=str, default="unc", choices=["unc", "umd", "google"])

    parser.add_argument(
        '--mask-choice',
        type=str,
        required=True,
        choices=[
            "best",
            "random",
            "first",
            "red_ellipse",
            "rectangle",
            "red_dense_mask",
            "reverse_blur"
        ]
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    original_name = args.original_name
    new_name = args.new_name
    dataset = args.dataset
    dataset_root = args.dataset_root
    data_split = args.split
    test_on_small_subset = not args.full
    img_size = 480

    original_pseudo_mask_folder = f"segment_stage_masks/{original_name}/"
    if data_split != "train":
        original_pseudo_mask_folder = f"segment_stage_masks_{data_split}/{original_name}/"
    
    new_chosen_mask_folder = f"select_stage_masks/{new_name}/"
    if data_split != "train":
        new_chosen_mask_folder = f"select_stage_masks_{data_split}/{new_name}/"

    if dataset != "refcocog":
        original_pseudo_mask_folder += f"instance_masks/{dataset}"
        new_chosen_mask_folder += f"instance_masks/{dataset}"
    else:
        original_pseudo_mask_folder += f"instance_masks/{dataset}_{args.split_by}"
        new_chosen_mask_folder += f"instance_masks/{dataset}_{args.split_by}"

    if not os.path.exists(new_chosen_mask_folder):
        os.makedirs(new_chosen_mask_folder)

    old_dataset, num_classes = get_dataset(
        dataset,
        dataset_root,
        data_split,
        split_by=args.split_by
    )
    coco_images_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), old_dataset.refer.IMAGE_DIR)

    # testing on a small subset to determine whether the generated masks are good enough or not
    if test_on_small_subset:
        print("Testing on 1000 examples...")
        n_examples = 1000
    else:
        print("Creating the full dataset...")
        n_examples = len(old_dataset)

    if args.mask_choice not in ["best", "random", "first"]:
        print("loading CLIP...")
        device = "cuda:0"
        clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)
        print("loaded")

    for i in tqdm(range(n_examples)):
        dataset_object = old_dataset[i]

        ref_id = dataset_object[-1]['ref_id']
        gt_mask = np.array(dataset_object[1])

        pseudo_mask_img_name = os.path.join(original_pseudo_mask_folder, f"pseudo_gt_mask_{ref_id}.png")
        instance_merged_pseudo_mask = np.array(Image.open(pseudo_mask_img_name).convert('L'))

        all_instance_masks = separate_instance_masks(instance_merged_pseudo_mask)

        if len(all_instance_masks) == 0:
            chosen_mask = instance_merged_pseudo_mask
        else:
            if args.mask_choice == "best":
                chosen_mask = best_mask_choice(all_instance_masks, gt_mask)
            elif args.mask_choice == "random":
                chosen_mask = random_mask_choice(dataset_object[0], all_instance_masks)
            elif args.mask_choice == "first":
                chosen_mask = first_mask_choice(dataset_object[0], all_instance_masks)
            else:
                chosen_mask = vp_and_max_sim_clip_choice(
                    clip_model,
                    preprocess,
                    dataset_object[0],
                    all_instance_masks,
                    dataset_object[-1]['sentences_sent'],
                    method=args.mask_choice
                )

                # couldn't decide on a mask, return an empty one to ignore this example in training
                if chosen_mask is None:
                    chosen_mask = np.zeros_like(instance_merged_pseudo_mask)

        save_binary_object_mask(
            chosen_mask,
            ref_id,
            new_chosen_mask_folder
        )
