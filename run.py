import utils
import argparser
import os
import time

import numpy as np
import random
import torch
from torch.utils import data
from torch import distributed
from torch.utils.data.distributed import DistributedSampler

from dataset import get_dataset
from metrics import StreamSegMetrics
from train import Trainer
from utils.logger import Logger


def save_ckpt(path, model, epoch):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict()
    }
    torch.save(state, path)


def log_val(logger, val_metrics, val_score, val_loss, cur_epoch):
    logger.info(val_metrics.to_str(val_score))

    # visualize validation score and samples
    logger.add_scalar("V-Loss", val_loss, cur_epoch)
    logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
    logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
    logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
    logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch)
    # logger.add_figure("Val_Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)


def log_samples(logger, ret_samples, denorm, label2color, cur_epoch):
    for k, (img, target, pred) in enumerate(ret_samples):
        img = (denorm(img) * 255).astype(np.uint8)
        target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
        pred = label2color(pred).transpose(2, 0, 1).astype(np.uint8)

        concat_img = np.concatenate((img, target, pred), axis=2)  # concat along width
        logger.add_image(f'Sample_{k}', concat_img, cur_epoch)


def main(opts):
    # ===== Setup distributed =====
    distributed.init_process_group(backend='nccl', init_method='env://')
    if opts.device is not None:
        device_id = opts.device
    else:
        device_id = opts.local_rank
    device = torch.device(device_id)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    if opts.device is not None:
        torch.cuda.set_device(opts.device)
    else:
        torch.cuda.set_device(device_id)

    # ===== Initialize logging =====
    logdir_full = f"{opts.logdir}/{opts.dataset}/{opts.name}/"
    if rank == 0:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize)
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)

    logger.print(f"Device: {device}")

    checkpoint_path = f"checkpoints/{opts.dataset}/{opts.name}.pth"
    os.makedirs(f"checkpoints/{opts.dataset}", exist_ok=True)

    # ===== Setup random seed to reproducibility =====
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # ===== Set up dataset =====
    train_dst, val_dst = get_dataset(opts, train=True)

    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size,
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True, pin_memory=True)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)
    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, "
                f"Val set: {len(val_dst)}, n_classes {opts.num_classes}")
    logger.info(f"Total batch size is {opts.batch_size * world_size}")
    # This is necessary for computing the scheduler decay
    opts.max_iter = opts.max_iter = opts.epochs * len(train_loader)

    # ===== Set up model and ckpt =====
    model = Trainer(device, logger, opts)
    model.distribute()

    cur_epoch = 0
    if opts.continue_ckpt:
        opts.ckpt = checkpoint_path
    if opts.ckpt is not None:
        assert os.path.isfile(opts.ckpt), "Error, ckpt not found. Check the correct directory"
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        cur_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state"])
        logger.info("[!] Model restored from %s" % opts.ckpt)
        del checkpoint
    else:
        logger.info("[!] Train from scratch")

    # ===== Train procedure =====
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    # uncomment if you want qualitative on val
    # if rank == 0 and opts.sample_num > 0:
    #     sample_ids = np.random.choice(len(val_loader), opts.sample_num, replace=False)  # sample idxs for visualization
    #     logger.info(f"The samples id are {sample_ids}")
    # else:
    #     sample_ids = None

    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    train_metrics = StreamSegMetrics(opts.num_classes)
    val_metrics = StreamSegMetrics(opts.num_classes)
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))

    while cur_epoch < opts.epochs and not opts.test:
        # =====  Train  =====
        start = time.time()
        epoch_loss = model.train(cur_epoch=cur_epoch, train_loader=train_loader,
                                 metrics=train_metrics, print_int=opts.print_interval)
        train_score = train_metrics.get_results()
        end = time.time()

        len_ep = int(end - start)
        logger.info(f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0] + epoch_loss[1]:.4f}, "
                    f"Class Loss={epoch_loss[0]:.4f}, Reg Loss={epoch_loss[1]}\n"
                    f"Train_Acc={train_score['Overall Acc']:.4f}, Train_Iou={train_score['Mean IoU']:.4f} "
                    f"\n -- time: {len_ep // 60}:{len_ep % 60} -- ")
        logger.info(f"I will finish in {len_ep * (opts.epochs - cur_epoch) // 60} minutes")

        logger.add_scalar("E-Loss", epoch_loss[0] + epoch_loss[1], cur_epoch)
        # logger.add_scalar("E-Loss-reg", epoch_loss[1], cur_epoch)
        # logger.add_scalar("E-Loss-cls", epoch_loss[0], cur_epoch)

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0:
            logger.info("validate on val set...")
            val_loss, _ = model.validate(loader=val_loader, metrics=val_metrics, ret_samples_ids=None)
            val_score = val_metrics.get_results()

            logger.print("Done validation")
            logger.info(f"End of Validation {cur_epoch}/{opts.epochs}, Validation Loss={val_loss}")

            log_val(logger, val_metrics, val_score, val_loss, cur_epoch)

            # keep the metric to print them at the end of training
            results["V-IoU"] = val_score['Class IoU']
            results["V-Acc"] = val_score['Class Acc']

        # =====  Save Model  =====
        if rank == 0:
            if not opts.debug:
                save_ckpt(checkpoint_path, model, cur_epoch)
                logger.info("[!] Checkpoint saved.")

        cur_epoch += 1

    torch.distributed.barrier()

    # ==== TESTING =====
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_dst = get_dataset(opts, train=False)
    test_loader = data.DataLoader(test_dst, batch_size=opts.batch_size_test,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)

    if rank == 0 and opts.sample_num > 0:
        sample_ids = np.random.choice(len(test_loader), opts.sample_num, replace=False)  # sample idxs for visual.
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    val_loss, ret_samples = model.validate(loader=test_loader, metrics=val_metrics, ret_samples_ids=sample_ids)
    val_score = val_metrics.get_results()
    conf_matrixes = val_metrics.get_conf_matrixes()
    logger.print("Done test on all")
    logger.info(f"*** End of Test on all, Total Loss={val_loss}")

    logger.info(val_metrics.to_str(val_score))
    log_samples(logger, ret_samples, denorm, label2color, 0)

    logger.add_figure("Test_Confusion_Matrix_Recall", conf_matrixes['Confusion Matrix'])
    logger.add_figure("Test_Confusion_Matrix_Precision", conf_matrixes["Confusion Matrix Pred"])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    results["T-Prec"] = val_score['Class Prec']
    logger.add_results(results)
    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'])
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'])
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'])
    ret = val_score['Mean IoU']

    logger.close()
    return ret


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    main(opts)
