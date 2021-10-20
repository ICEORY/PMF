import torch
import tensorboardX
import time
from option import Option
import torch.nn as nn
import datetime
import pc_processor
from pc_processor.checkpoint import Recorder
import numpy as np


class Trainer(object):
    def __init__(self, settings: Option, model: nn.Module, recorder=None):
        # init params
        self.settings = settings
        self.recorder = recorder
        self.model = model.cuda()
        self.remain_time = pc_processor.utils.RemainTime(
            self.settings.n_epochs)

        # init data loader
        if self.settings.distributed:
            self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self._initDataloader()
        else:
            self.train_loader, self.val_loader = self._initDataloader()
        # init criterion
        self.criterion = self._initCriterion()
        # init optimizer
        self.optimizer = self._initOptimizer()

        # set multi gpu
        if self.settings.n_gpus > 1:
            if self.settings.distributed:
                self.model = pc_processor.layers.sync_bn.replaceBN(
                    self.model).cuda()
                self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.settings.gpu])
            else:
                self.model = nn.DataParallel(self.model)
                # repalce bn with sync_bn
                self.model = pc_processor.layers.sync_bn.replaceBN(
                    self.model).cuda()
                for k, v in self.criterion.items():
                    self.criterion[k] = nn.DataParallel(v).cuda()
        # get metrics
        self.metrics = pc_processor.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device("cpu"), ignore=self.ignore_class)
        self.metrics.reset()

        # warmup cosine lr
        self.scheduler = pc_processor.utils.WarmupCosineLR(
            optimizer=self.optimizer,
            lr=self.settings.lr,
            warmup_steps=self.settings.warmup_epochs * len(self.train_loader),
            momentum=self.settings.momentum,
            max_steps=len(self.train_loader) * (self.settings.n_epochs - self.settings.warmup_epochs))

    def _initOptimizer(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.settings.lr)
        return optimizer

    def _initDataloader(self):
        # add NuScenes dataloader
        if self.settings.dataset == "nuScenes":
            print('----Using nuScenes dataset----')
            trainset = pc_processor.dataset.nuScenes.Nuscenes(
                root=self.settings.data_root, version='v1.0-trainval',
                split='train', return_ref=False, has_image=False)
            valset = pc_processor.dataset.nuScenes.Nuscenes(
                root=self.settings.data_root, version='v1.0-trainval',
                split='val', return_ref=False, has_image=False)
            self.mapped_cls_name = trainset.mapped_cls_name
            self.ignore_class = [0]
            self.cls_weight = np.ones((self.settings.n_classes))
            self.cls_weight[0] = 0

        elif self.settings.dataset == "SemanticKitti":
            data_config_path = "../../pc_processor/dataset/semantic_kitti/semantic-kitti.yaml"
            trainset = pc_processor.dataset.semantic_kitti.SemanticKitti(
                root=self.settings.data_root,
                sequences=[0,1,2,3,4,5,6,7,9,10],
                config_path=data_config_path
            )
            self.cls_weight = 1 / (trainset.cls_freq + 1e-3)
            self.ignore_class = []
            for cl, _ in enumerate(self.cls_weight):
                if trainset.data_config["learning_ignore"][cl]:
                    self.cls_weight[cl] = 0
                if self.cls_weight[cl] < 1e-10:
                    self.ignore_class.append(cl)
            if self.recorder is not None:
                self.recorder.logger.info("weight: {}".format(self.cls_weight))
            self.mapped_cls_name = trainset.mapped_cls_name

            valset = pc_processor.dataset.semantic_kitti.SemanticKitti(
                root=self.settings.data_root,
                sequences=[8],
                config_path=data_config_path
            )

        else:
            raise ValueError(
                "invalid dataset: {}".format(self.settings.dataset))

        train_salsa_loader = pc_processor.dataset.SalsaNextLoader(
            dataset=trainset,
            config=self.settings.config)

        val_salsa_loader = pc_processor.dataset.SalsaNextLoader(
            dataset=valset,
            config=self.settings.config, 
            is_train=False)

        if self.settings.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
            train_loader = torch.utils.data.DataLoader(
                train_salsa_loader,
                batch_size=self.settings.batch_size,
                num_workers=self.settings.n_threads,
                drop_last=True,
                sampler=train_sampler
            )

            val_loader = torch.utils.data.DataLoader(
                val_salsa_loader,
                batch_size=self.settings.batch_size,
                num_workers=self.settings.n_threads,
                drop_last=False,
                sampler=val_sampler
            )
            return train_loader, val_loader, train_sampler, val_sampler

        else:
            train_loader = torch.utils.data.DataLoader(
                train_salsa_loader,
                batch_size=self.settings.batch_size,
                num_workers=self.settings.n_threads,
                shuffle=True,
                drop_last=True)

            val_loader = torch.utils.data.DataLoader(
                val_salsa_loader,
                batch_size=self.settings.batch_size,
                num_workers=self.settings.n_threads,
                shuffle=False,
                drop_last=False
            )
            return train_loader, val_loader, None, None

    def _initCriterion(self):
        criterion = {}
        criterion["lovasz"] = pc_processor.loss.Lovasz_softmax(ignore=0)
        if self.settings.dataset == "SemanticKitti":
            alpha = np.log(1+self.cls_weight)
            alpha = alpha / alpha.max()
        elif self.settings.dataset == "nuScenes":
            alpha = np.ones((self.settings.n_classes))
        alpha[0] = 0
        if self.recorder is not None:
            self.recorder.logger.info("focal_loss alpha: {}".format(alpha))
        criterion["focal_loss"] = pc_processor.loss.FocalSoftmaxLoss(
            self.settings.n_classes, gamma=2, alpha=alpha, softmax=False)

        # set device
        for _, v in criterion.items():
            v.cuda()
        return criterion

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self, epoch, mode="Train"):
        if mode == "Train":
            dataloader = self.train_loader
            self.model.train()
            if self.settings.distributed:
                self.train_sampler.set_epoch(epoch)
        elif mode == "Validation":
            dataloader = self.val_loader
            self.model.eval()
        else:
            raise ValueError("invalid mode: {}".format(mode))

        # init metrics meter
        loss_meter = pc_processor.utils.AverageMeter()
        self.metrics.reset()
        total_iter = len(dataloader)
        t_start = time.time()
        for i, (input_feature, input_label, input_mask) in enumerate(dataloader):

            t_process_start = time.time()

            input_feature = input_feature.cuda()
            input_label = input_label.cuda().long()
            input_label = input_label * input_label.ge(1).long()
            input_mask = input_mask.cuda() * input_label.ge(1).float()

            # forward propergation
            if mode == "Train":
                output = self.model(input_feature)
                loss_s = self.criterion["focal_loss"](
                    output, input_label, mask=input_mask)
                loss_lovasz = self.criterion["lovasz"](
                    output, input_label)
                total_loss = loss_lovasz + loss_s
                # backward
                if self.settings.n_gpus > 1:
                    total_loss = total_loss.mean()
                self._backward(total_loss)
            else:
                with torch.no_grad():
                    output = self.model(input_feature)
                    loss_s = self.criterion["focal_loss"](
                        output, input_label, mask=input_mask)  
                    loss_lovasz = self.criterion["lovasz"](
                        output, input_label)
                        
                    total_loss = loss_lovasz + loss_s
                    if self.settings.n_gpus > 1:
                        total_loss = total_loss.mean()

            # measure accuracy and record loss
            loss = total_loss.mean()
            with torch.no_grad():
                argmax = output.argmax(dim=1)
                self.metrics.addBatch(argmax, input_label)
                mean_acc, class_acc = self.metrics.getAcc()
                mean_iou, class_iou = self.metrics.getIoU()
                mean_recall, class_recall = self.metrics.getRecall()
                # sync distributed tensor
                if self.settings.distributed:
                    mean_acc = mean_acc.cuda()
                    mean_iou = mean_iou.cuda()
                    mean_recall = mean_recall.cuda()

                    torch.distributed.barrier()
                    torch.distributed.all_reduce(mean_acc)
                    torch.distributed.all_reduce(mean_iou)
                    torch.distributed.all_reduce(mean_recall)

                    mean_acc = mean_acc.cpu() / self.settings.world_size
                    mean_iou = mean_iou.cpu() / self.settings.world_size
                    mean_recall = mean_recall.cpu() / self.settings.world_size

            loss_meter.update(loss.item(), input_feature.size(0))

            # timer logger ----------------------------------------
            t_process_end = time.time()
            data_cost_time = t_process_start - t_start
            process_cost_time = t_process_end - t_process_start
            self.remain_time.update(cost_time=(time.time() - t_start), mode=mode)
            remain_time = datetime.timedelta(
                seconds=self.remain_time.getRemainTime(
                    epoch=epoch, iters=i, total_iter=total_iter, mode=mode
                ))
            t_start = time.time()

            if self.recorder is not None:
                for g in self.optimizer.param_groups:
                    lr = g["lr"]
                    break
                log_str = ">>> {} E[{:03d}|{:03d}] I[{:04d}|{:04d}] DT[{:.3f}] PT[{:.3f}] ".format(
                    mode, self.settings.n_epochs, epoch + 1, total_iter, i, data_cost_time, process_cost_time)
                log_str += "LR {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} ".format(
                    lr, loss.item(), mean_acc.item(), mean_iou.item())
                log_str += "RT {}".format(remain_time)
                self.recorder.logger.info(log_str)
            if mode == "Train":
                self.scheduler.step()

            if self.settings.is_debug:
                break
        # tensorboard logger
        if self.recorder is not None:
            self.recorder.tensorboard.add_scalar(
                tag="{}_Loss".format(mode), scalar_value=loss_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_LossSoftmax".format(mode), scalar_value=loss_s.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_LossLovasz".format(mode), scalar_value=loss_lovasz.item(), global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_meanAcc".format(mode), scalar_value=mean_acc.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_meanIOU".format(mode), scalar_value=mean_iou.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_meanRecall".format(mode), scalar_value=mean_recall.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_lr".format(mode), scalar_value=lr, global_step=epoch)

            # -----------------
            for i, (_, v) in enumerate(self.mapped_cls_name.items()):
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_Acc".format(mode, i, v), scalar_value=class_acc[i].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_Recall".format(mode, i, v), scalar_value=class_recall[i].item(),
                    global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_IOU".format(mode, i, v), scalar_value=class_iou[i].item(), global_step=epoch)

            for i in range(output.size(1)):
                self.recorder.tensorboard.add_image(
                    "{}_Pred_cls_{:02d}".format(mode, i), output[0, i:i + 1].cpu(), epoch)
            for i in range(output.size(1)):
                self.recorder.tensorboard.add_image("{}_Label_cls_{:02d}".format(
                    mode, i), input_label[0:1].eq(i).cpu(), epoch)
            for i in range(input_feature.size(1)):
                self.recorder.tensorboard.add_image("{}_Inputs_{}".format(mode, i), input_feature[0,i:i+1].cpu(), epoch)

            log_str = ">>> {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} Recall {:0.4f}".format(
                mode, loss_meter.avg, mean_acc.item(), mean_iou.item(), mean_recall.item())
            self.recorder.logger.info(log_str)

        result_metrics = {
            "Acc": mean_acc.item(),
            "IOU": mean_iou.item(),
            "Recall": mean_recall.item()
        }

        return result_metrics
