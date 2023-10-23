import argparse
from PIL import Image

import torch
import matplotlib.pyplot as plt

from ssc_ris.refer_dataset.bert.modeling_bert import BertModel
from ssc_ris.refer_dataset.bert.tokenization_bert import BertTokenizer
from ssc_ris.refer_dataset.utils import get_transform
from ssc_ris.utils.lavt_lib import segmentation


def get_parser():
    parser = argparse.ArgumentParser(description="LAVT training and testing")
    parser.add_argument(
        "--model-checkpoint", required=True, help="model from checkpoint"
    )
    parser.add_argument("--input-image", required=True, help="input image")
    parser.add_argument("--sentence", required=True, help="RIS sentence")
    parser.add_argument("--device", default="cuda:0", help="device")

    # model parameters
    parser.add_argument(
        "--fusion_drop", default=0.0, type=float, help="dropout rate for PWAMs"
    )
    parser.add_argument(
        "--mha",
        default="",
        help="If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,"
        "where a, b, c, and d refer to the numbers of heads in stage-1,"
        "stage-2, stage-3, and stage-4 PWAMs",
    )
    parser.add_argument(
        "--swin_type",
        default="base",
        help="tiny, small, base, or large variants of the Swin Transformer",
    )
    parser.add_argument(
        "--window12",
        action="store_true",
        help="only needs specified when testing,"
        "when training, window size is inferred from pre-trained weights file name"
        "(containing 'window12'). Initialize Swin with window size 12 instead of the default 7.",
    )

    return parser


def plot_side_by_side(img, mask, sentence):
    _, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(img)
    axs[0].set_axis_off()

    axs[1].imshow(img)
    axs[1].imshow(mask, alpha=0.6)
    axs[1].set_title(sentence)
    axs[1].set_axis_off()

    plt.tight_layout()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint
    sentence_raw = args.sentence
    img_path = args.input_image
    device = args.device

    img_size = 480
    bert_model_name = "bert-base-uncased"
    args.window12 = True

    # load the model
    single_model = segmentation.__dict__["lavt"](pretrained="", args=args)
    checkpoint = torch.load(model_checkpoint, map_location="cpu")
    single_model.load_state_dict(checkpoint["model"])
    model = single_model.to(device)

    model_class = BertModel
    single_bert_model = model_class.from_pretrained(bert_model_name)
    single_bert_model.pooler = None

    single_bert_model.load_state_dict(checkpoint["bert_model"])
    bert_model = single_bert_model.to(device)

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    attention_mask = [0] * 20
    padded_input_ids = [0] * 20
    input_ids = tokenizer.encode(text=sentence_raw, add_special_tokens=True)

    # truncation of tokens
    input_ids = input_ids[:20]
    padded_input_ids[: len(input_ids)] = input_ids
    attention_mask[: len(input_ids)] = [1] * len(input_ids)

    sentence = torch.tensor(padded_input_ids).unsqueeze(0).to(device)
    attention = torch.tensor(attention_mask).unsqueeze(0).to(device)

    # load image and perform the input transformations
    orig_img = Image.open(img_path).convert("RGB")
    transform = get_transform(img_size=img_size)
    img, _ = transform(orig_img, orig_img)
    img = img.unsqueeze(0).to(device)

    # inference
    last_hidden_states = bert_model(sentence, attention_mask=attention)[0]
    embedding = last_hidden_states.permute(0, 2, 1)
    output = model(img, embedding, l_mask=attention.unsqueeze(-1))
    output_mask = output.cpu().argmax(1).data.numpy()

    # plot and save output
    plot_side_by_side(
        orig_img.resize((img_size, img_size)), output_mask[0], sentence_raw
    )
    plt.savefig("example_output.png")
