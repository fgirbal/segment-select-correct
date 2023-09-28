import datetime
import os
import gc
import time
from functools import reduce
import operator
from typing import Tuple, Callable
from timeit import default_timer as timer

import wandb
import numpy as np
import torch
import torch.utils.data

from ssc_ris.refer_dataset.bert.modeling_bert import BertModel
from ssc_ris.refer_dataset.bert.tokenization_bert import BertTokenizer
from ssc_ris.refer_dataset.utils import get_dataset, get_unsupervised_dataset, get_transform
from ssc_ris.utils.lavt_lib import segmentation
import ssc_ris.utils.lavt_lib.lavt_utils as utils
import ssc_ris.correct.loss as loss
from ssc_ris.utils import IoU, wandb_mask

from train_test_args import get_parser


# Should only be used for quicker testing
log_on_wandb = False
os.environ["WANDB__SERVICE_WAIT"] = "300"


def evaluate(
            model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            bert_model: BertModel
            ) -> Tuple[torch.tensor, torch.tensor]:
    """Evaluate a model on the data loader provided. Returns mIoU and oIoU.

    Args:
        model (torch.nn.Module): model to be evaluated
        data_loader (torch.utils.data.DataLoader): val/test dataloader
        bert_model (BertModel): Bert model

    Returns:
        Tuple[torch.tensor, torch.tensor]: (mIoU, oIoU)
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    # wandb masks
    all_wandb_masks = []
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                output = model(image, embedding, l_mask=attentions)
            else:
                output = model(image, sentences, l_mask=attentions)

            if ((total_its % 100) == 0 and model.device_ids[0] == 0 and log_on_wandb):
                all_wandb_masks.extend([
                    wandb_mask(
                        image=img.cpu().transpose(1,2).transpose(0,2).numpy(),
                        sentence=sentence.cpu(),
                        pred_mask=output_mask.cpu().detach().numpy(),
                        true_mask=tar.cpu().numpy(),
                        matched_mask=None,
                        bert_model=bert_model
                    )
                    for img, sentence, output_mask, tar in zip(image, sentences, output.argmax(1), target)
                ])

            iou, I, U = IoU(output, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    if log_on_wandb:
        wandb.log({
            "eval mIoU": mIoU * 100.0,
            "eval oIoU": cum_I * 100.0 / cum_U
        })

        wandb.log({"validation predictions": all_wandb_masks})

    return 100 * iou, 100 * cum_I / cum_U


def train_one_epoch(
            model: torch.nn.Module,
            loss_fn: Callable,
            optimizer: torch.optim.Optimizer,
            data_loader: torch.utils.data.DataLoader,
            lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
            epoch: int,
            print_freq: int,
            iterations: int,
            bert_model: BertModel,
            batch_size_limit: int = 18,
            contrastive_alpha: float = 0.01
            ) -> None:
    """Train one epoch of our stage 3 using loss_fn.

    Args:
        model (torch.nn.Module): trainable model
        loss_fn (Callable): loss function to be used
        optimizer (torch.optim.Optimizer): optimizer to be used
        data_loader (torch.utils.data.DataLoader): training data
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): scheduler
        epoch (int): current epoch
        print_freq (int): printing frequency for loggin
        iterations (int): number of current iterations
        bert_model (BertModel): Bert model used for processing text input
        batch_size_limit (int, optional): upper limit of the batch size. Defaults to 18.
        contrastive_alpha (float, optional): if using contrastive training (Appendix of the original paper),
            this sets that parameter. Defaults to 0.01.
    """
    # Fixed choice of order per epoch for reproducibility
    torch.manual_seed(100)

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    params = {"contrastive_alpha": contrastive_alpha}

    start = timer()

    buffer_images = None
    buffer_targets = None
    buffer_sentences = None
    buffer_attentions = None
    buffer_mask_idx_info = None
    for data in metric_logger.log_every(data_loader, print_freq, header, skip_first=True):
        images, targets, sentences, attentions, _ = data

        # unfold the image, mask target and sentences into individual inputs for the model
        unfolded_images = []
        unfolded_targets = []
        unfolded_sentences = []
        unfolded_attentions = []
        unfolded_mask_idx_info = []
        if buffer_images is not None:
            unfolded_images = [buffer_images]
            unfolded_targets = [buffer_targets]
            unfolded_sentences = [buffer_sentences]
            unfolded_attentions = [buffer_attentions]
            unfolded_mask_idx_info = [buffer_mask_idx_info]

        for batch_idx, image in enumerate(images):
            for mask_idx, mask in enumerate(targets[batch_idx]):
                # ignore empty masks, or masks that cover the full image
                if (mask == 0).all() or (mask == 1).all():
                    continue

                image_mask_sentences = sentences[batch_idx][mask_idx]
                image_mask_attentions = attentions[batch_idx][mask_idx]

                repeated_mask = mask.repeat(image_mask_sentences.shape[0], 1, 1)
                repeated_image = image.repeat(image_mask_sentences.shape[0], 1, 1, 1)

                unfolded_images.append(repeated_image)
                unfolded_targets.append(repeated_mask)
                unfolded_sentences.append(image_mask_sentences)
                unfolded_attentions.append(image_mask_attentions)
                unfolded_mask_idx_info.append(torch.tensor(mask_idx).repeat(image_mask_sentences.shape[0]))

        # if there are no images in the buffer so far
        if len(unfolded_images) == 0:
            del data
            gc.collect()

            continue

        buffer_images = torch.cat(unfolded_images)
        buffer_targets = torch.cat(unfolded_targets)
        buffer_sentences = torch.cat(unfolded_sentences)
        buffer_attentions = torch.cat(unfolded_attentions)
        buffer_mask_idx_info = torch.cat(unfolded_mask_idx_info)

        # perform batches while the buffer contains at least batch_size_limit points
        while buffer_images.shape[0] > batch_size_limit:
            total_its += 1

            # get a bacth of batch_size_limit images
            batch_images = buffer_images[:batch_size_limit]
            batch_targets = buffer_targets[:batch_size_limit]
            batch_sentences = buffer_sentences[:batch_size_limit]
            batch_attentions = buffer_attentions[:batch_size_limit]
            batch_mask_idx_info = buffer_mask_idx_info[:batch_size_limit]

            # remove them from the buffer
            buffer_images = buffer_images[batch_size_limit:]
            buffer_targets = buffer_targets[batch_size_limit:]
            buffer_sentences = buffer_sentences[batch_size_limit:]
            buffer_attentions = buffer_attentions[batch_size_limit:]
            buffer_mask_idx_info = buffer_mask_idx_info[batch_size_limit:]

            # process them
            batch_images, batch_targets, batch_sentences, batch_attentions = batch_images.cuda(non_blocking=True),\
                                                batch_targets.cuda(non_blocking=True),\
                                                batch_sentences.cuda(non_blocking=True),\
                                                batch_attentions.cuda(non_blocking=True)

            if bert_model is not None:
                last_hidden_states = bert_model(batch_sentences, attention_mask=batch_attentions)[0]  # (6, 10, 768)
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                batch_attentions = batch_attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
                output = model(batch_images, embedding, l_mask=batch_attentions)
            else:
                output = model(batch_images, batch_sentences, l_mask=batch_attentions)

            # instance cross entropy term
            loss, loss_log = loss_fn(
                batch_images,
                output,
                batch_targets,
                batch_mask_idx_info,
                params=params
            )

            optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            torch.cuda.synchronize()
            train_loss += loss.item()
            iterations += 1
            if len(loss_log) == 0: 
                metric_logger.update(
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]["lr"]
                )
                if log_on_wandb:
                    wandb.log({
                        "batch loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "batch iter": total_its
                    })
            else:
                metric_logger.update(
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]["lr"],
                    ice_loss=loss_log["ice loss"],
                    contrastive_loss=loss_log["contrastive loss"]
                )
                if log_on_wandb:
                    wandb.log({
                        "batch loss": loss.item(),
                        "batch ice loss": loss_log["ice loss"],
                        "batch contrastive loss": loss_log["contrastive loss"],
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "batch iter": total_its
                    })

            if (total_its % 250) == 1 and model.device_ids[0] == 0 and log_on_wandb:
                if "matched masks" not in loss_log:
                    mask_list = [
                        wandb_mask(
                            image=image,
                            sentence=sentence,
                            pred_mask=output_mask,
                            true_mask=batch_pseudo_gt_mask,
                            matched_mask=None,
                            bert_model=bert_model
                        )
                        for image, sentence, output_mask, batch_pseudo_gt_mask in zip(
                            batch_images.transpose(2,3).transpose(1,3).cpu().numpy(),
                            batch_sentences,
                            output.detach().argmax(1).cpu().numpy(),
                            batch_targets.cpu().numpy()
                        )
                    ]
                else:
                    matched_masks = loss_log["matched masks"]
                    mask_list = [
                        wandb_mask(
                            image=image,
                            sentence=sentence,
                            pred_mask=output_mask,
                            true_mask=batch_pseudo_gt_mask,
                            matched_mask=matched_mask,
                            bert_model=bert_model
                        )
                        for image, sentence, output_mask, batch_pseudo_gt_mask, matched_mask in zip(
                            batch_images.transpose(2,3).transpose(1,3).cpu().numpy(),
                            batch_sentences,
                            output.detach().argmax(1).cpu().numpy(),
                            batch_targets.cpu().numpy(),
                            matched_masks.cpu().numpy()
                        )
                    ]

                wandb.log({"training predictions": mask_list, "epoch": epoch})

            del batch_images, batch_targets, batch_sentences, batch_attentions, loss, output
            if bert_model is not None:
                del last_hidden_states, embedding

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # garbage collect to avoid memory leaks
        del data, unfolded_images, unfolded_targets, unfolded_sentences, unfolded_attentions
        gc.collect()

    end = timer()
    if model.device_ids[0] == 0:
        print(f"Epoch training time: {end - start:.2f}")

    if log_on_wandb:
        wandb.log({
            "loss": train_loss / total_its,
            "lr": optimizer.param_groups[0]["lr"]
        })


def main(args):
    dataset, _ = get_unsupervised_dataset(
        dataset=args.dataset,
        dataset_root=args.refer_data_root,
        pseudo_masks_root=args.pseudo_masks_root,
        data_split="train",
        transforms=get_transform(img_size=args.img_size),
        split_by=args.split_by,
        one_sentence=args.one_sentence_per_item
    )
    
    dataset_test, _ = get_dataset(
        dataset=args.dataset,
        dataset_root=args.refer_data_root,
        data_split="val",
        transforms=get_transform(img_size=args.img_size),
        split_by=args.split_by,
        return_attributes=False
    )

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True
    )
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True,
        collate_fn=dataset.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers
    )

    # model initialization
    model = segmentation.__dict__["lavt"](
        pretrained=args.pretrained_swin_weights,
        args=args
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module

    model_class = BertModel
    bert_model = model_class.from_pretrained(args.ck_bert)
    bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
    bert_model.cuda()
    bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
    bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
    single_bert_model = bert_model.module

    # add the tokenizer to the bert_model object to be able to decode embeddings
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    bert_model.tokenizer = bert_tokenizer

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        single_bert_model.load_state_dict(checkpoint['bert_model'])

    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        single_bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
        # the following are the parameters of bert
        {"params": reduce(operator.concat,
                            [[p for p in single_bert_model.encoder.layer[i].parameters()
                            if p.requires_grad] for i in range(10)])},
    ]

    # optimizer
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9
    )

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    if log_on_wandb:
        wandb.init(
            project="ssc-ris-experiments",
            group=f"{args.model_experiment_name}",
            config={
                "model-experiment-name": args.model_experiment_name,
                "dataset": args.dataset,
                "one-sentence-per-item": args.one_sentence_per_item,
                "num-processes": num_tasks,
                "batch-size-unfolded": args.batch_size_unfolded_limit,
                "initial-learning-rate": args.lr,
                "contrastive-alpha": args.contrastive_alpha,
                "epochs": args.epochs,
                "pseudo-masks-root": args.pseudo_masks_root,
                "output-dir": args.output_dir,
                "groundtruth-masks": args.groundtruth_masks
            }
        )

    if args.loss_mode == "random_ce":
        loss_fn = loss.random_assign_and_cross_entropy
    elif args.loss_mode == "greedy_ce":
        loss_fn = loss.greedy_match_and_cross_entropy
    elif args.loss_mode == "greedy_ce_contrastive":
        loss_fn = loss.greedy_match_and_contrastive
    else:
        raise NotImplemented

    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(
            model,
            loss_fn,
            optimizer,
            data_loader,
            lr_scheduler,
            epoch,
            args.print_freq,
            iterations,
            bert_model,
            batch_size_limit=args.batch_size_unfolded_limit,
            contrastive_alpha=args.contrastive_alpha
        )
        iou, overallIoU = evaluate(model, data_loader_test, bert_model)

        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))
            if single_bert_model is not None:
                dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}
            else:
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(
                dict_to_save,
                os.path.join(args.output_dir, '{}_best_{}.pth'.format(args.model_experiment_name, args.model_id))
            )
            best_oIoU = overallIoU

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)

    main(args)
