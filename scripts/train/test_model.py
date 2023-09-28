import os
from typing import Tuple

import numpy as np
import torch
import torch.utils.data

from ssc_ris.refer_dataset.bert.modeling_bert import BertModel
from ssc_ris.refer_dataset.utils import get_dataset, get_transform
from ssc_ris.utils.lavt_lib import segmentation
import ssc_ris.utils.lavt_lib.lavt_utils as utils

from train_test_args import get_parser


def evaluate(
            model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            bert_model: BertModel,
            device: torch.device,
            model_name: str,
            split_name: str
            ):
    """Evaluate a model on the data loader provided. Prints and writes results to file.

    Args:
        model (torch.nn.Module): model to be evaluated
        data_loader (torch.utils.data.DataLoader): val/test dataloader
        bert_model (BertModel): Bert model
        device (torch.device): device where to perform the computations
        model_name (str): model identifier for results
        split_name (str): dataset split identifier to file writing
    """
    def computeIoU(pred_seg: np.ndarray, gd_seg: np.ndarray) -> Tuple[float, float]:
        I = np.sum(np.logical_and(pred_seg, gd_seg))
        U = np.sum(np.logical_or(pred_seg, gd_seg))

        return I, U

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del image, target, sentences, attentions, output, output_mask
            if bert_model is not None:
                del last_hidden_states, embedding

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    results_str = 'Final results:\nMean IoU is %.2f\n' % (mIoU*100.)
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    if not os.path.exists("results/"):
        os.makedirs("results/")

    with open(f"results/{model_name[:-4]}_{split_name}.txt", 'w') as f:
        f.write(results_str)


def main(args):
    device = torch.device(args.device)

    # load the dataset and prep the dataloader
    dataset_test, _ = get_dataset(
        dataset=args.dataset,
        dataset_root=args.refer_data_root,
        data_split=args.split,
        transforms=get_transform(img_size=args.img_size),
        split_by=args.split_by,
        return_attributes=False,
        eval_model=True
    )
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers
    )

    # load the model
    single_model = segmentation.__dict__["lavt"](pretrained='',args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    model_class = BertModel
    single_bert_model = model_class.from_pretrained(args.ck_bert)
    # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
    if args.ddp_trained_weights:
        single_bert_model.pooler = None
    
    single_bert_model.load_state_dict(checkpoint['bert_model'])
    bert_model = single_bert_model.to(device)

    evaluate(
        model,
        data_loader_test,
        bert_model,
        device=device,
        model_name=os.path.basename(args.resume),
        split_name=args.split
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)
