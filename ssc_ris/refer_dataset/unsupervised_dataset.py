import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image

from .bert.tokenization_bert import BertTokenizer
from .refer.refer import REFER


class UnsupervisedReferDataset(data.Dataset):
    def __init__(
                self,
                dataset: str,
                splitBy: str = 'unc',
                refer_data_root: str = 'dataset/refer',
                pseudo_masks_data_root='outputs',
                image_transforms: object = None,
                target_transforms: object = None,
                split: str = 'train',
                bert_tokenizer: str = 'bert-base-uncased',
                COCO_image_root: str = None,
                eval_mode: bool = False,
                groundtruth_masks=False,
                one_sentence=False
            ):
        """Wrapper for the REFER class in refer (https://github.com/lichengunc/refer) for the case of unsupervised data
        (potentially including multiple masks per image).

        Args:
            dataset (str): dataset description (one of "refcoco", "refcoco+" or "refcocog")
            splitBy (str, optional): data split. Defaults to 'unc'.
            refer_data_root (str, optional): root of the REFER data. Defaults to 'dataset/refer'.
            pseudo_masks_data_root (str, optional): file location where the unsupervised masks are located. Defaults to 'outputs'.
            image_transforms (object, optional): transformations to be applied to input images. Defaults to None.
            target_transforms (object, optional): transformations to be applied to target masks. Defaults to None.
            split (str, optional): dataset split. Defaults to 'train'.
            bert_tokenizer (str, optional): type of BERT tokenizer. Defaults to 'bert-base-uncased'.
            COCO_image_root (str, optional): image root. Defaults to None.
            eval_mode (bool, optional): if True, load all sentences, otherwise chose only one. Defaults to False.
            return_attributes (bool, optional): whether additional data attributes should be returned. Defaults to False.
        """

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(refer_data_root, dataset, splitBy, image_root=COCO_image_root)
        self.pseudo_masks_data_root = os.path.join(pseudo_masks_data_root, dataset)
        if dataset == "refcocog":
            self.pseudo_masks_data_root += f"_{splitBy}"

        self.groundtruth_masks = groundtruth_masks
        self.one_sentence = one_sentence

        if groundtruth_masks:
            print("WARNING: you are using UnsupervisedReferDataset with groundtruth masks, which should only be used for testing...")

        self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = {}
        self.attention_masks = {}
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['sent']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids[ref['ref_id']] = torch.vstack(sentences_for_ref)
            self.attention_masks[ref['ref_id']] = torch.vstack(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        this_img = self.imgs[index]
        this_img_id = this_img['id']

        # get the ids of the references corresponding to this image
        possible_refs = self.refer.imgToRefs[this_img_id]
        refs = [ref for ref in possible_refs if ref['ref_id'] in self.ref_ids]
        PIL_img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")
        if self.image_transforms is None:
            img = PIL_img

        target_per_ref = []
        tensor_embeddings_per_ref = []
        attention_mask_per_ref = []
        attributes_per_ref = []
        for ref in refs:
            ref_id = ref['ref_id']
            
            if self.groundtruth_masks:
                # groundtruth masks
                ref_mask = np.array(self.refer.getMask(ref)['mask'])
                annot = np.zeros(ref_mask.shape)
                annot[ref_mask == 1] = 1

                annot = Image.fromarray(annot.astype(np.uint8), mode="P")
            else:
                # pseudo-gt masks
                annot = Image.open(os.path.join(self.pseudo_masks_data_root, f"pseudo_gt_mask_{ref_id}.png")).convert("P")
            
            target = annot

            if self.image_transforms is not None:
                # resize, from PIL to tensor, and mean and std normalization
                img, target = self.image_transforms(PIL_img, annot)

            # embedding = []
            # att = []
            # for e, a in zip(self.input_ids[ref_id], self.attention_masks[ref_id]):
            #     embedding.append(e.unsqueeze(-1))
            #     att.append(a.unsqueeze(-1))

            if not self.one_sentence:
                tensor_embeddings = self.input_ids[ref_id]
                attention_mask = self.attention_masks[ref_id]
            else:
                choice_sent = np.random.choice(len(self.input_ids[ref_id]))
                tensor_embeddings = self.input_ids[ref_id][choice_sent:choice_sent+1]
                attention_mask = self.attention_masks[ref_id][choice_sent:choice_sent+1]

            target_per_ref.append(target)
            tensor_embeddings_per_ref.append(tensor_embeddings)
            attention_mask_per_ref.append(attention_mask)
            attributes_per_ref.append({
                "sentence_ids": [sent["sent_id"] for sent in ref["sentences"]],
                "sentences_raw": [sent["raw"] for sent in ref["sentences"]],
                "sentences_sent": [sent["sent"] for sent in ref["sentences"]],
                "ref_id": ref["ref_id"],
                "ann_id": ref["ann_id"],
                "ref": ref
            })

        return img, target_per_ref, tensor_embeddings_per_ref, attention_mask_per_ref, attributes_per_ref

    @staticmethod
    def collate_fn(data_items):
        return tuple(zip(*data_items))
