import torch
from torch import distributed
import torch.nn as nn
from segmentation_module import make_model
from utils.scheduler import get_scheduler
from apex.parallel import DistributedDataParallel
from apex import amp
from utils.loss import HardNegativeMining, MeanReduction


class Trainer:
    def __init__(self, device, logger, opts):
        self.logger = logger
        self.device = device
        self.opts = opts

        self.model = make_model(opts)
        self.model = self.model.to(device)
        self.distributed = False

        logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")

        logger.debug(self.model)

        # xxx Set up optimizer
        params = []
        params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters()),
                       'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                       'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.parameters()),
                       'weight_decay': opts.weight_decay})

        self.optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)
        self.scheduler = get_scheduler(opts, self.optimizer)
        logger.debug("Optimizer:\n%s" % self.optimizer)

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opts.opt_level)

        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if opts.hnm else MeanReduction()

    def distribute(self):
        if self.model is not None:
            # Put the model on GPU
            self.distributed = True
            self.model = DistributedDataParallel(self.model, delay_allreduce=True)

    def train(self, cur_epoch, train_loader, metrics=None, print_int=10):
        """Train and return epoch loss"""
        device = self.device
        model = self.model
        optim = self.optimizer
        scheduler = self.scheduler
        logger = self.logger

        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        l_reg = torch.tensor(0.)

        train_loader.sampler.set_epoch(cur_epoch)

        model.train()
        for cur_step, (images, labels) in enumerate(train_loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optim.zero_grad()
            outputs = model(images)

            # xxx Cross Entropy Loss
            loss = self.reduction(self.criterion(outputs, labels), labels)  # B x H x W
            # loss = self.criterion(outputs, labels)

            loss_tot = loss

            with amp.scale_loss(loss_tot, optim) as scaled_loss:
                scaled_loss.backward()

            optim.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            interval_loss += loss.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            _, prediction = outputs.max(dim=1)  # B, H, W
            labels = labels.cpu().numpy()
            prediction = prediction.cpu().numpy()
            if metrics is not None:
                metrics.update(labels, prediction)

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                            f" Loss={interval_loss}")
                logger.debug(f"Loss made of: CE {loss}, LRec {l_reg}")
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss', interval_loss, x)
                interval_loss = 0.0

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
            reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)

        # collect statistics from multiple processes
        if metrics is not None:
            metrics.synch(device)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")

        return epoch_loss, reg_loss

    def validate(self, loader, metrics, ret_samples_ids=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        logger = self.logger
        model.eval()

        class_loss = 0.0

        ret_samples = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(images)

                loss = criterion(outputs, labels)

                class_loss += loss.item()

                _, prediction = outputs.max(dim=1)  # B, H, W
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(),
                                        labels[0], prediction[0]))

            # collect statistics from multiple processes
            metrics.synch(device)

            class_loss = torch.tensor(class_loss).to(self.device)
            torch.distributed.reduce(class_loss, dst=0)
            if distributed.get_rank() == 0:
                class_loss = class_loss / distributed.get_world_size() / len(loader)

            if logger is not None:
                logger.info(f"Validation, Class Loss={class_loss}")

        return class_loss, ret_samples

    def load_state_dict(self, checkpoint):
        state = {}
        if not self.distributed:
            for k, v in checkpoint["model"].items():
                state[k[7:]] = v

        model_state = state if not self.distributed else checkpoint['model']
        self.model.load_state_dict(model_state, strict=True)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict()}
        return state