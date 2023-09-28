import os
from typing import List
from collections import Counter

import cv2
import numpy as np
import nltk
import spacy
import torch
import clip
from segment_anything import SamPredictor
from groundingdino.util.inference import Model as GroundingDino

from ._utils import save_all_masks

nlp = spacy.load("en_core_web_md")
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

def get_nouns_and_noun_phrases_nltk(text: str) -> List[str]:
    """Extract the key noun/noun phrases from text using NLTK.

    Args:
        text (str): input string

    Returns:
        List[str]: list of nouns/noun phrases
    """
    # extracting nouns is easy
    def get_nouns(pos_tags):
        tags = ['NN', 'NNS', 'NNP', 'VBG', 'VB', 'VBD', 'VBN', 'VBP', 'VBZ']
        ret = []
        for (word, pos) in pos_tags:
            if pos in tags:
                ret.append(word)

        return ret
    
    # somehow this affects NLTK quite a lot
    if text[-1] != ".":
        text += "."

    tokenized = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokenized)
    # nouns = get_nouns(pos_tags)

    #  noun phrases...
    # Taken from Su Nam Kim Paper...
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)
    chunked = chunker.parse(pos_tags)
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if isinstance(subtree, nltk.Tree):
            current_chunk.append(' '.join([token for token, _ in subtree.leaves()]))
        elif current_chunk:
            named_entity = ' '.join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    
    if current_chunk:
        continuous_chunk.extend(current_chunk)

    return continuous_chunk


def get_nouns_and_noun_phrases_spacy(text: str) -> List[str]:
    """Extract the key noun/noun phrases from text using NLTK.

    Args:
        text (str): input string

    Returns:
        List[str]: list of nouns/noun phrases
    """
    doc = nlp(text)
    # add "the" at the beginning of the sentence to help with the parsing
    if doc[0].text != 'the':
        doc = nlp('the ' + text)

    subj_nouns = []
    for token in doc:
        if (token.pos_ == "NOUN" or token.pos_ == "PROPN") and (token.dep_ == "ROOT" or token.dep_ == "nsubj"):
            subj_nouns.append(token.text)

    if len(subj_nouns) != 0:
        return subj_nouns
    
    return [token.text for token in doc if token.pos_ == "NOUN"]


def project_to_dataset_classes(
            clip_model: torch.nn.Module,
            noun_list: List[str],
            dataset_object_list: List[str],
            dataset_object_list_features: torch.Tensor,
            with_context: bool = False,
            top_k: int = 1,
            context_string: str = "a photo of a "
            ) -> List[str]:
    """Function that uses a CLIP text embeddings to determine which COCO object classes are closest
    to the nouns passed in this list.

    Args:
        clip_model (torch.nn.Module): CLIP model to use for projection
        noun_list (List[str]): list of nouns to project
        dataset_object_list (List[str]): list of objects in the dataset to detect
        dataset_object_features (torch.Tensor): CLIP features of the dataset object obtained using clip_model
        with_context (bool, optional): whether a contextual string should be used. Defaults to False.

    Returns:
        List[str]: list of projected nouns
    """
    device = next(clip_model.parameters()).device

    ret_COCO_objs_list = []
    for noun_phrase in noun_list:
        with torch.no_grad():
            if with_context:
                noun_embedding = clip_model.encode_text(
                    clip.tokenize(context_string + noun_phrase).to(device)
                )
            else:
                noun_embedding = clip_model.encode_text(
                    clip.tokenize(noun_phrase).to(device)
                )
        
        noun_embedding = noun_embedding.to(torch.float32)
        noun_embedding /= noun_embedding.norm(dim=-1, keepdim=True)

        similarities = noun_embedding @ dataset_object_list_features.T

        if top_k == 1:
            closest_COCO_obj = dataset_object_list[similarities[0].argmax().cpu().numpy()]
            ret_COCO_objs_list.append(closest_COCO_obj)
        else:
            closest_COCO_objs = list(dataset_object_list[similarities[0].topk(top_k).indices.cpu().numpy()])
            ret_COCO_objs_list.extend(closest_COCO_objs)
    
    return ret_COCO_objs_list


def enhance_class_name(class_names: List[str]) -> List[str]:
    # function copied from the GroundingDINO repository
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


def segment_from_detections(
            sam_predictor: SamPredictor,
            image: np.ndarray,
            xyxy: np.ndarray
            ) -> np.ndarray:
    """Use SAM to segment an image given an array of detection bounding boxes xyxy.

    Args:
        sam_predictor (SamPredictor): SAM prediction model
        image (np.ndarray): input image
        xyxy (np.ndarray): detection bounding boxes

    Returns:
        np.ndarray: resulting masks
    """
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        # take the most likely mask
        index = np.argmax(scores)
        result_masks.append(masks[index])

    return np.array(result_masks)


def segment_from_image_and_nouns(
            grounding_dino_model: GroundingDino,
            sam_predictor: SamPredictor,
            image_name: str,
            image_path: str,
            nouns: List[List[str]],
            save_path: str = None,
            sentences: List[List[str]] = None,
            one_query_per_noun: bool = False,
            most_likely_noun: bool = False
            ):
    """Segment an image given a sequence of nouns. Based on GroundingSAM
    (https://github.com/IDEA-Research/Grounded-Segment-Anything).

    Args:
        grounding_dino_model (GroundingDino): GroundingDino model
        sam_predictor (SamPredictor): SAM prediction model
        image_name (str): name of the image to segment
        image_path (str): path to the image folder
        nouns (List[List[str]]): list of nouns per sentence and per image to segment in the image
        save_path (str, optional): if passed, images will be saved. Defaults to None.
        sentences (List[List[str]], optional): list of sentences per reference for visualization.
            Defaults to None.
        one_query_per_noun (bool, optional): if True, GroundingDino is called with each name individually as opposed to all of them as a set.
            This tends to generate more false positive masks. Defaults to False.
        most_likely_noun (bool, optional): if True, will only query GroundingDino with the most likely noun out of the possible ones.
            Defaults to False.

    Returns:
        _type_: _description_
    """
    full_noun_set = []
    nouns_per_object = []
    for obj_nouns in nouns:
        nouns_for_obj = []
        for sent_nouns in obj_nouns:
            nouns_for_obj.extend(sent_nouns)

        nouns_per_object.append(nouns_for_obj)

        if most_likely_noun:
            # choose the most frequent object as the one to focus on
            if len(nouns_for_obj) != 0:
                # likely_obj = max(set(nouns_for_obj), key = nouns_for_obj.count)
                likely_obj = Counter(nouns_for_obj).most_common(1)[0][0]
            else:
                likely_obj = "object"
            
            full_noun_set.extend([likely_obj])
        else:
            full_noun_set.extend(nouns_for_obj)

    if len(full_noun_set) == 0:
        full_noun_set = np.array(['object'])

    full_noun_set = list(set(full_noun_set))
    full_image_path = os.path.join(image_path, image_name)
    image = cv2.imread(full_image_path)

    if not one_query_per_noun:
        # perform one query to the model including all the nouns in the sentences
        detection_classes = []
        try:
            # detect using grouding dino
            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=full_noun_set,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            detection_classes = detections.class_id
        except:
            detections = None

        if detections is None or None in detections.class_id:
            try:
                # detect using grouding dino
                detections = grounding_dino_model.predict_with_classes(
                    image=image,
                    classes=enhance_class_name(full_noun_set),
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD
                )
                detection_classes = detections.class_id
            except:
                print("----- failed to detect anything ----")
                detections = None
        
        masks = []
        if detections is not None and len(detections) != 0:
            # segment using SAM
            masks = segment_from_detections(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )
    else:
        # perform one query per noun in the sentences; leads to more possible masks
        detections_xyxy = []
        detection_classes = []
        for noun_id, noun_set_noun in enumerate(full_noun_set):
            try:
                # detect using grouding dino
                dets = grounding_dino_model.predict_with_classes(
                    image=image,
                    classes=[noun_set_noun],
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD
                )
            except:
                dets = None

            if dets is not None:
                detections_xyxy.append(dets.xyxy)
                detection_classes.append(dets.class_id + noun_id)

        masks = []
        if len(detections_xyxy) != 0:
            detections_xyxy = np.concatenate(detections_xyxy)
            detection_classes = np.concatenate(detection_classes)

            # segment using SAM
            masks = segment_from_detections(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections_xyxy
            )

    # instance segmentation masks
    object_instance_masks = []
    for obj_nouns in nouns_per_object:
        instance_object_mask = np.zeros(image.shape[:2])

        instance_index = 1
        for mask, class_id in zip(masks, detection_classes):
            if class_id is None:
                continue
            
            class_name = full_noun_set[class_id]
            if class_name in obj_nouns:
                instance_object_mask[mask] = instance_index
                instance_index += 1
        
        object_instance_masks.append(instance_object_mask)

    if save_path:
        save_all_masks(full_image_path, object_instance_masks, sentences, output_name=save_path)

    return object_instance_masks
