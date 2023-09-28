# Segment: Stage 1 of the three-stage framework presented 
import os
import json
import inspect
import pickle
import argparse

from tqdm import tqdm
import numpy as np
import torch
import clip
import groundingdino
from groundingdino.util.inference import Model as GroundingDino
from segment_anything import sam_model_registry, SamPredictor

from ssc_ris.refer_dataset.utils import get_dataset, save_binary_object_masks
from ssc_ris.segment import get_nouns_and_noun_phrases_nltk, get_nouns_and_noun_phrases_spacy, segment_from_image_and_nouns, project_to_dataset_classes


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-f', '--full', action='store_true')
    
    # dataset arguments
    parser.add_argument('-d', '--dataset', type=str, default="refcoco")
    parser.add_argument('--dataset-root', type=str, default="refer/data")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--split-by', type=str, default="unc", choices=["unc", "umd", "google"])
    parser.add_argument('-s', '--start-index', type=int, default=0)

    # text projection and segmentation
    parser.add_argument('--bg-threshold', type=float, default=0.95)
    parser.add_argument('--project', action='store_true')
    parser.add_argument('--keep-top-k-matches', type=int, default=1)
    parser.add_argument('--context-projections', action='store_true')
    parser.add_argument('--noun-extraction', type=str, choices=["nltk", "spacy"], default="spacy")
    parser.add_argument(
        '--one-query-per-noun',
        action='store_true',
        help="if true, will call the segmentation model once for each noun in the sentences; leads to more false positives"
    )
    parser.add_argument(
        '--most-likely-noun',
        action='store_true',
        help="query only using the most likely noun in the set of sentences"
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
    bg_threshold = args.bg_threshold
    keep_top_k_matches = args.keep_top_k_matches
    project_to_COCO_classes = args.project
    with_context_projections = args.context_projections
    noun_extraction = args.noun_extraction
    test_on_small_subset = not args.full

    context_string = "a photo of a "
    COCO_obj_list = np.array([
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if project_to_COCO_classes:
        clip_model, preprocess = clip.load("ViT-B/32", device=device)

        if not with_context_projections:
            if os.path.isfile("CLIP_COCO_obj_list_features.pb"):
                with open("CLIP_COCO_obj_list_features.pb", "rb") as f:
                    CLIP_COCO_obj_list_features = pickle.load(f).to(device)
            else:
                with torch.no_grad():
                    CLIP_COCO_obj_list_features = torch.vstack([
                        clip_model.encode_text(clip.tokenize(obj_name).to(device)) for obj_name in COCO_obj_list
                    ])

                CLIP_COCO_obj_list_features = CLIP_COCO_obj_list_features.to(torch.float32)
                CLIP_COCO_obj_list_features /= CLIP_COCO_obj_list_features.norm(dim=-1, keepdim=True)

                with open("CLIP_COCO_obj_list_features.pb", "wb") as f:
                    pickle.dump(CLIP_COCO_obj_list_features, f)
        else:
            if os.path.isfile("CLIP_COCO_obj_list_features_w_context.pb"):
                with open("CLIP_COCO_obj_list_features_w_context.pb", "rb") as f:
                    CLIP_COCO_obj_list_features = pickle.load(f).to(device)
            else:
                with torch.no_grad():
                    CLIP_COCO_obj_list_features = torch.vstack([
                        clip_model.encode_text(clip.tokenize(context_string + obj_name).to(device)) for obj_name in COCO_obj_list
                    ])

                CLIP_COCO_obj_list_features = CLIP_COCO_obj_list_features.to(torch.float32)
                CLIP_COCO_obj_list_features /= CLIP_COCO_obj_list_features.norm(dim=-1, keepdim=True)

                with open("CLIP_COCO_obj_list_features_w_context.pb", "wb") as f:
                    pickle.dump(CLIP_COCO_obj_list_features, f)

    print("loading GroundingDino...")
    grounding_dino_root = os.path.dirname(groundingdino.__file__)
    grounding_dino_model = GroundingDino(
        model_config_path=os.path.join(grounding_dino_root, "config/GroundingDINO_SwinT_OGC.py"),
        model_checkpoint_path=os.path.join(grounding_dino_root, "../weights/groundingdino_swint_ogc.pth")
    )
    print("loaded GroundingDino")

    print("loading SAM...")
    sam = sam_model_registry["vit_h"](
        checkpoint=os.path.join(grounding_dino_root, "../../sam_weights/sam_vit_h_4b8939.pth")
    ).to(device=device)
    sam_predictor = SamPredictor(sam)
    print("loaded SAM")

    experiment_folder = f"segment_stage_masks/{experiment_name}/"
    if data_split != "train":
        experiment_folder = f"segment_stage_masks_{data_split}/{experiment_name}/"

    output_mask_folder = experiment_folder
    if dataset != "refcocog":
        output_mask_folder += f"instance_masks/{dataset}"
    else:
        output_mask_folder += f"instance_masks/{dataset}_{args.split_by}"

    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)

    masks_viz_save_path = f"segment_stage_masks/{experiment_name}/masks_viz/{dataset}"
    if not os.path.exists(masks_viz_save_path):
        os.makedirs(masks_viz_save_path)

    old_dataset, num_classes = get_dataset(
        dataset,
        dataset_root,
        data_split,
        split_by=args.split_by
    )
    coco_images_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), old_dataset.refer.IMAGE_DIR)

    # write the configuration file to be able to trace which parameters were used in generating this data
    with open(experiment_folder + f"generation_details_{dataset}_{args.split_by}.log", 'w') as f:
        json.dump(vars(args), f, indent=4)

    # testing on a small subset to determine whether the generated masks are good enough or not
    if test_on_small_subset:
        print("Testing on 1000 examples...")
        n_examples = 1000
    else:
        print("Creating the full dataset...")
        n_examples = len(old_dataset)

    if args.start_index != 0:
        print(f"Starting mask generation from index {args.start_index} of the dataset...")

    current_image = ""
    image_sentences = []
    image_nouns = []
    ref_ids = []
    save_path = None
    for i in tqdm(range(n_examples)):
        if i < args.start_index:
            continue

        dataset_object = old_dataset[i]

        if current_image == "":
            # this is the first image ever, just write it down and move on
            current_image = dataset_object[-1]['file_name']
        elif dataset_object[-1]['file_name'] != current_image:
            # it's a new image; process the info in the noun buffer and clear it to prepare for new ones
            if test_on_small_subset:
                save_path = os.path.join(masks_viz_save_path, f"mask_{i}.png")

            object_masks = segment_from_image_and_nouns(
                grounding_dino_model,
                sam_predictor,
                current_image,
                coco_images_directory,
                image_nouns,
                save_path,
                image_sentences,
                one_query_per_noun=args.one_query_per_noun,
                most_likely_noun=args.most_likely_noun
            )
            save_binary_object_masks(
                object_masks,
                ref_ids,
                output_mask_folder
            )

            current_image = dataset_object[-1]['file_name']
            image_sentences = []
            image_nouns = []
            ref_ids = []

        obj_nouns = []
        for sentence in dataset_object[-1]['sentences_sent']:
            if noun_extraction == "nltk":
                nouns = get_nouns_and_noun_phrases_nltk(sentence)
                if len(nouns) == 0:
                    nouns = get_nouns_and_noun_phrases_spacy(sentence)
            elif noun_extraction == "spacy":
                nouns = get_nouns_and_noun_phrases_spacy(sentence)
                if len(nouns) == 0:
                    nouns = get_nouns_and_noun_phrases_nltk(sentence)

            if project_to_COCO_classes:
                nouns = project_to_dataset_classes(
                    clip_model,
                    nouns,
                    dataset_object_list=COCO_obj_list,
                    dataset_object_list_features=CLIP_COCO_obj_list_features,
                    with_context=with_context_projections,
                    top_k=keep_top_k_matches
                )

            obj_nouns.append(nouns)

        image_sentences.append(dataset_object[-1]['sentences_sent'])
        image_nouns.append(obj_nouns)
        ref_ids.append(dataset_object[-1]['ref_id'])

    object_masks = segment_from_image_and_nouns(
        grounding_dino_model,
        sam_predictor,
        current_image,
        coco_images_directory,
        image_nouns,
        save_path=None,
        sentences=image_sentences,
        one_query_per_noun=args.one_query_per_noun,
        most_likely_noun=args.most_likely_noun
    )
    save_binary_object_masks(
        object_masks,
        ref_ids,
        output_mask_folder
    )
