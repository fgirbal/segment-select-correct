import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image

from .bert.tokenization_bert import BertTokenizer
from .refer.refer import REFER


class ReferDataset(data.Dataset):
    def __init__(
                self,
                dataset: str,
                splitBy: str = 'unc',
                refer_data_root: str = 'dataset/refer',
                image_transforms: object = None,
                target_transforms: object = None,
                split: str = 'train',
                bert_tokenizer: str = 'bert-base-uncased',
                COCO_image_root: str = None,
                eval_mode: bool = False,
                return_attributes: bool = False
            ):
        """Wrapper for the REFER class in refer (https://github.com/lichengunc/refer)

        Args:
            dataset (str): dataset description (one of "refcoco", "refcoco+" or "refcocog")
            splitBy (str, optional): data split. Defaults to 'unc'.
            refer_data_root (str, optional): root of the REFER data. Defaults to 'dataset/refer'.
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
        self.return_attributes = return_attributes

        self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
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

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")
        target = annot

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, target = self.image_transforms(img, annot)

        if self.eval_mode:
            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        if self.return_attributes:
            attributes = {
                "file_name": this_img["file_name"],
                "sentence_ids": [sent["sent_id"] for sent in ref[0]["sentences"]],
                "sentences_raw": [sent["raw"] for sent in ref[0]["sentences"]],
                "sentences_sent": [sent["sent"] for sent in ref[0]["sentences"]],
                "ref_id": ref[0]["ref_id"],
                "ann_id": ref[0]["ann_id"],
                "ref": ref[0]
            }

            return img, target, tensor_embeddings, attention_mask, attributes
        else:
            return img, target, tensor_embeddings, attention_mask
