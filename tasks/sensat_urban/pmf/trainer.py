import numpy as np
import os
import torch
import time
from option import Option
import torch.nn as nn
import datetime
import pc_processor
from pc_processor.checkpoint import Recorder
import math
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, settings: Option, model: nn.Module, recorder=None):
        # init params
        self.settings = settings
        self.recorder = recorder
        self.model = model.cuda()
        self.remain_time = pc_processor.utils.RemainTime(
            self.settings.n_epochs)

        # init data loader
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self._initDataloader()

        # init criterion
        self.criterion = self._initCriterion()
    
        # init optimizer

        [self.optimizer, self.aux_optimizer] = self._initOptimizer()

        # set multi gpu
        if self.settings.n_gpus > 1:
            if self.settings.distributed:
                # sync bn
                self.model = pc_processor.layers.sync_bn.replaceBN(
                    self.model).cuda()
                self.model = nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.settings.gpu]) #  find_unused_parameters=True

            else:
                self.model = nn.DataParallel(self.model)
                # repalce bn with sync_bn
                self.model = pc_processor.layers.sync_bn.replaceBN(
                    self.model).cuda()
                for k, v in self.criterion.items():
                    self.criterion[k] = nn.DataParallel(v).cuda()

        # get metrics for pcd
        self.metrics = pc_processor.metrics.IOUEval(
            n_classes=self.settings.nclasses, device=torch.device("cpu"),
            ignore=self.ignore_class, is_distributed=self.settings.distributed)
        self.metrics.reset()

        # get metrics for img
        self.metrics_img = pc_processor.metrics.IOUEval(
            n_classes=self.settings.nclasses, device=torch.device("cpu"),
            ignore=self.ignore_class, is_distributed=self.settings.distributed)
        self.metrics_img.reset()

        self.scheduler = pc_processor.utils.WarmupCosineLR(
            optimizer=self.optimizer,
            lr=self.settings.lr,
            warmup_steps=self.settings.warmup_epochs * len(self.train_loader),
            momentum=self.settings.momentum,
            max_steps=len(self.train_loader) * (self.settings.n_epochs-self.settings.warmup_epochs))

        self.aux_scheduler = pc_processor.utils.WarmupCosineLR(
            optimizer=self.aux_optimizer,
            lr=self.settings.lr,
            warmup_steps=self.settings.warmup_epochs * len(self.train_loader),
            momentum=self.settings.momentum,
            max_steps=len(self.train_loader) * (self.settings.n_epochs-self.settings.warmup_epochs))
        
    # ------------------------------------------------------------------
    # functions for initialization
    # ------------------------------------------------------------------
    def _initOptimizer(self):
        # check params
        adam_params = [{"params": self.model.lidar_stream.parameters()}]

        adam_opt = torch.optim.AdamW(
            params=adam_params, lr=self.settings.lr, amsgrad=True)

        sgd_params = [
            {"params": self.model.camera_stream_encoder.parameters()},
            {"params": self.model.camera_stream_decoder.parameters()}]

        sgd_opt = torch.optim.SGD(
            params=sgd_params, lr=self.settings.lr,
            nesterov=True,
            momentum=self.settings.momentum,
            weight_decay=self.settings.weight_decay)
        optimizer = [adam_opt, sgd_opt]

        return optimizer

    def _initDataloader(self):
        if self.settings.dataset == "SensatUrban":
            trainset = pc_processor.dataset.SensatUrban(
                root_path=self.settings.data_root,
                split="train",
                keep_idx=False
            )
            self.ignore_class = [0]
            
            self.mapped_cls_name = trainset.mapped_cls_name

            valset = pc_processor.dataset.SensatUrban(
                root_path=self.settings.data_root,
                split="val",
                keep_idx=False,
                use_crop=True,
            )
        else:
            raise ValueError(
                "invalid dataset: {}".format(self.settings.dataset))
        
        trainset_loader = pc_processor.dataset.SensatLoader(
            dataset=trainset,
            img_h=self.settings.img_h, img_w=self.settings.img_w,
            n_samples_split=self.settings.n_samples_split)

        valset_loader = pc_processor.dataset.SensatLoader(
            dataset=valset)

        if self.settings.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset_loader, shuffle=True, drop_last=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                valset_loader, shuffle=False, drop_last=False)
            train_loader = torch.utils.data.DataLoader(
                trainset_loader,
                batch_size=self.settings.batch_size[0],
                num_workers=self.settings.n_threads,
                drop_last=True,
                sampler=train_sampler
            )
            val_loader = torch.utils.data.DataLoader(
                valset_loader,
                batch_size=self.settings.batch_size[1],
                num_workers=self.settings.n_threads,
                drop_last=False,
                sampler=val_sampler
            )
            

            return train_loader, val_loader, train_sampler, val_sampler

        else:
            train_loader = torch.utils.data.DataLoader(
                trainset_loader,
                batch_size=self.settings.batch_size[0],
                num_workers=self.settings.n_threads,
                shuffle=True,
                drop_last=True)

            val_loader = torch.utils.data.DataLoader(
                valset_loader,
                batch_size=self.settings.batch_size[1],
                num_workers=self.settings.n_threads,
                shuffle=False,
                drop_last=False
            )
            return train_loader, val_loader, None, None

    def _initCriterion(self):
        criterion = {}
        criterion["lovasz"] = pc_processor.loss.Lovasz_softmax(ignore=0)

        criterion["kl_loss"] = nn.KLDivLoss(reduction="none")

        if self.settings.dataset == "SensatUrban":
            alpha = np.ones((self.settings.nclasses))
        alpha[0] = 0
        alpha[4] = 2
        alpha[5] = 2.5
        alpha[7] = 3
        alpha[12] = 10
        alpha[13] = 2.5

        if self.recorder is not None:
            self.recorder.logger.info("focal_loss alpha: {}".format(alpha))
        criterion["focal_loss"] = pc_processor.loss.FocalSoftmaxLoss(
            self.settings.nclasses, gamma=2, alpha=alpha, softmax=False)
       
        criterion["dice_loss"] = pc_processor.loss.ExpLogDiceLoss()

        # set device
        for _, v in criterion.items():
            v.cuda()
        return criterion

    # -------------------------------------------------------------------------
    # functions for running
    # -------------------------------------------------------------------------
    def _backward(self, loss):
        self.optimizer.zero_grad()
        self.aux_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.aux_optimizer.step()

    def _computeClassifyLoss(self, pred, label, label_mask):

        loss_foc = self.criterion["focal_loss"](
            pred, label, mask=label_mask)
        
        loss_dc = self.criterion["dice_loss"](pred, label, mask=label_mask)
        loss_foc = loss_foc + loss_dc

        loss_lovasz = self.criterion["lovasz"](
            pred, label)
               
        return loss_lovasz, loss_foc

    def _computePerceptionAwareLoss(self, pcd_entropy, img_entropy,
                             pcd_pred, pcd_pred_log, img_pred, img_pred_log):
        
        pcd_confidence = 1 - pcd_entropy
        img_confidence = 1 - img_entropy
        information_importance = pcd_confidence - img_confidence
        pcd_guide_mask = pcd_confidence.ge(0.7).float()
        img_guide_mask = img_confidence.ge(0.7).float()

        pcd_guide_weight = information_importance.gt(0).float(
        ) * information_importance.abs() * pcd_guide_mask 
        img_guide_weight = information_importance.lt(0).float(
        ) * information_importance.abs() * img_guide_mask

        # compute kl loss
        loss_per_pcd = (self.criterion["kl_loss"](
            pcd_pred_log, img_pred) * img_guide_weight.unsqueeze(1)).mean()
        loss_per_img = (self.criterion["kl_loss"](
            img_pred_log, pcd_pred) * pcd_guide_weight.unsqueeze(1)).mean()
        loss_per = loss_per_pcd + loss_per_img
        return loss_per, pcd_guide_weight, img_guide_weight

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
        loss_focal_meter = pc_processor.utils.AverageMeter()
        loss_lovasz_meter = pc_processor.utils.AverageMeter()
        entropy_meter = pc_processor.utils.AverageMeter()
        self.metrics.reset()

        loss_img_focal_meter = pc_processor.utils.AverageMeter()
        loss_img_lovasz_meter = pc_processor.utils.AverageMeter()
        entropy_img_meter = pc_processor.utils.AverageMeter()
        self.metrics_img.reset()

        loss_perception_meter = pc_processor.utils.AverageMeter()

        total_iter = len(dataloader)
        t_start = time.time()

        feature_mean = torch.Tensor(self.settings.feature_mean).unsqueeze(
            0).unsqueeze(2).unsqueeze(2).cuda()
        feature_std = torch.Tensor(self.settings.feature_std).unsqueeze(
            0).unsqueeze(2).unsqueeze(2).cuda()
        
        for i, (input_feature, input_label) in enumerate(dataloader):
            t_process_start = time.time()
            input_feature = input_feature.cuda()
            input_mask = input_feature[:, 4]
            # normalize input_feature
            input_feature = (input_feature - feature_mean)/feature_std * input_mask.unsqueeze(1)

            pcd_feature = input_feature[:, 0:5]
            img_feature = input_feature[:, 5:8]
            input_label = (input_label.cuda().long() + 1) * input_mask.long()
            label_mask = input_label.gt(0)

            # forward propergation
            if mode == "Train":
                lidar_pred, camera_pred = self.model(pcd_feature, img_feature)

                lidar_pred_log = torch.log(lidar_pred.clamp(min=1e-8))
                # compute pcd entropy: p * log p
                pcd_entropy = -(lidar_pred * lidar_pred_log).sum(1) / \
                    math.log(self.settings.nclasses)

                loss_lov, loss_foc = self._computeClassifyLoss(
                    pred=lidar_pred, label=input_label, label_mask=label_mask)

                # compute img entropy
                camera_pred_log = torch.log(
                    camera_pred.clamp(min=1e-8))
                # normalize to [0,1)
                img_entropy = - \
                    (camera_pred * camera_pred_log).sum(1) / \
                    math.log(self.settings.nclasses)

                loss_lov_cam, loss_foc_cam = self._computeClassifyLoss(
                    pred=camera_pred, label=input_label, label_mask=label_mask)

                loss_per, pcd_guide_weight, img_guide_weight = self._computePerceptionAwareLoss(
                    pcd_entropy=pcd_entropy, img_entropy=img_entropy,
                    pcd_pred=lidar_pred, pcd_pred_log=lidar_pred_log,
                    img_pred=camera_pred, img_pred_log=camera_pred_log
                )

                total_loss = loss_foc + loss_lov + \
                     loss_foc_cam + loss_lov_cam + \
                     loss_per * 0.5

                if self.settings.n_gpus > 1:
                    total_loss = total_loss.mean()
                
                loss_copy = total_loss.detach().clone()
                if self.settings.distributed:
                    torch.distributed.barrier()
                    torch.distributed.all_reduce(loss_copy)      
                if torch.isnan(loss_copy).sum() > 0:
                    if self.recorder is not None:
                        self.recorder.logger.info("loss is nan, stop bp...")
                    print("lovasz: ", loss_lov, loss_lov_cam)
                    print("loss_s: ", loss_foc, loss_foc_cam)
                    total_loss.backward()
                    self.optimizer.zero_grad()
                    self.aux_optimizer.zero_grad()
                    continue
                # backward
                self._backward(total_loss)
                # update lr after backward (required by pytorch)
                self.scheduler.step()
                self.aux_scheduler.step()

            else:
                with torch.no_grad():
                    lidar_pred, camera_pred = self.model(pcd_feature, img_feature)

                    lidar_pred_log = torch.log(lidar_pred.clamp(min=1e-8))
                    # compute pcd entropy: p * log p
                    pcd_entropy = -(lidar_pred * lidar_pred_log).sum(1) / \
                        math.log(self.settings.nclasses)

                    loss_lov, loss_foc = self._computeClassifyLoss(
                        pred=lidar_pred, label=input_label, label_mask=label_mask)

                    # compute img entropy
                    camera_pred_log = torch.log(
                        camera_pred.clamp(min=1e-8))
                    # normalize to [0,1)
                    img_entropy = - \
                        (camera_pred * camera_pred_log).sum(1) / \
                        math.log(self.settings.nclasses)

                    loss_lov_cam, loss_foc_cam = self._computeClassifyLoss(
                        pred=camera_pred, label=input_label, label_mask=label_mask)

                    loss_per, pcd_guide_weight, img_guide_weight = self._computePerceptionAwareLoss(
                        pcd_entropy=pcd_entropy, img_entropy=img_entropy,
                        pcd_pred=lidar_pred, pcd_pred_log=lidar_pred_log,
                        img_pred=camera_pred, img_pred_log=camera_pred_log
                    )

                    total_loss = loss_foc + loss_lov + \
                        loss_foc_cam + loss_lov_cam + \
                        loss_per * 0.5

                    if self.settings.n_gpus > 1:
                        total_loss = total_loss.mean()

            # measure accuracy and record loss
            loss = total_loss.mean()

            # # check output
            # measure accuracy and record loss
            with torch.no_grad():
                # compute iou and acc
                argmax = lidar_pred.argmax(dim=1)
                self.metrics.addBatch(argmax, input_label)
                mean_iou, class_iou = self.metrics.getIoU()
                mean_acc, class_acc = self.metrics.getAcc()
                mean_recall, class_recall = self.metrics.getRecall()

                argmax_img = camera_pred.argmax(dim=1)
                self.metrics_img.addBatch(argmax_img, input_label)
                mean_iou_img, class_iou_img = self.metrics_img.getIoU()
                mean_acc_img, class_acc_img = self.metrics_img.getAcc()
                mean_recall_img, class_recall_img = self.metrics_img.getRecall()


            loss_meter.update(total_loss.item(), input_feature.size(0))
            loss_focal_meter.update(loss_foc.item(), input_feature.size(0))
            loss_lovasz_meter.update(loss_lov.item(), input_feature.size(0))
            entropy_meter.update(pcd_entropy.mean().item(), input_feature.size(0))

            loss_img_lovasz_meter.update(loss_lov_cam.item(), input_feature.size(0))
            loss_img_focal_meter.update(loss_foc_cam.item(), input_feature.size(0))
            entropy_img_meter.update(img_entropy.mean().item(), input_feature.size(0))

            loss_perception_meter.update(loss_per.item(), input_feature.size(0))

            # timer logger ----------------------------------------
            t_process_end = time.time()

            data_cost_time = t_process_start - t_start
            process_cost_time = t_process_end - t_process_start

            self.remain_time.update(cost_time=(time.time()-t_start), mode=mode)
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
                    mode, self.settings.n_epochs, epoch+1, total_iter, i+1, data_cost_time, process_cost_time)
                log_str += "LR {:0.5f} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} Recall {:0.4f} Entropy {:0.4f} ".format(
                    lr, loss.item(), mean_acc.item(), mean_iou.item(), mean_recall.item(), entropy_meter.avg)
                log_str += "ImgAcc {:0.4f} ImgIOU {:0.4F} ImgRecall {:0.4f} ImgEntropy {:0.4f} ".format(
                    mean_acc_img.item(), mean_iou_img.item(), mean_recall_img.item(), entropy_img_meter.avg)
                log_str += "RT {}".format(remain_time)
                self.recorder.logger.info(log_str)

            if self.settings.is_debug:
                break

        # tensorboard logger
        if self.recorder is not None:
            # scalar log
            self.recorder.tensorboard.add_scalar(
                tag="{}_Loss".format(mode), scalar_value=loss_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_LossFocal".format(mode), scalar_value=loss_focal_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_LossLovasz".format(mode), scalar_value=loss_lovasz_meter.avg, global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_lr".format(mode), scalar_value=lr, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_entropy".format(mode), scalar_value=entropy_meter.avg, global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_meanAcc".format(mode), scalar_value=mean_acc.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_meanIOU".format(mode), scalar_value=mean_iou.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_meanRecall".format(mode), scalar_value=mean_recall.item(), global_step=epoch)

            for i, (_, v) in enumerate(self.mapped_cls_name.items()):
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_Acc".format(mode, i, v), scalar_value=class_acc[i].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_Recall".format(mode, i, v), scalar_value=class_recall[i].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_IOU".format(mode, i, v), scalar_value=class_iou[i].item(), global_step=epoch)

            # record img branch acc, recall and iou
            self.recorder.tensorboard.add_scalar(
                tag="{}_LossImageFocal".format(mode), scalar_value=loss_img_focal_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_LossImageLovasz".format(mode), scalar_value=loss_img_lovasz_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_ImageEntropy".format(mode), scalar_value=entropy_img_meter.avg, global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_LossPerception".format(mode), scalar_value=loss_perception_meter.avg, global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_Image_meanAcc".format(mode), scalar_value=mean_acc_img.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_Image_meanIOU".format(mode), scalar_value=mean_iou_img.item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_Image_meanRecall".format(mode), scalar_value=mean_recall_img.item(), global_step=epoch)

            for i, (_, v) in enumerate(self.mapped_cls_name.items()):
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_ImageAcc".format(mode, i, v), scalar_value=class_acc_img[i].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_ImageRecall".format(mode, i, v), scalar_value=class_recall_img[i].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}_{}_ImageIOU".format(mode, i, v), scalar_value=class_iou_img[i].item(), global_step=epoch)

            if epoch % self.settings.print_frequency == 0 and self.settings.dataset != "nuScenes":
                # img log
                for i in range(1, pcd_feature.size(1)):
                    self.recorder.tensorboard.add_image(
                        "{}_PCDFeature_{}".format(mode, i-1), pcd_feature[0, i:i+1].cpu(), epoch)

                if camera_pred is not None:
                    for i in range(1, camera_pred.size(1)):
                        self.recorder.tensorboard.add_image(
                            "{}_RGBPred_cls_{:02d}_{}".format(mode, i-1, self.mapped_cls_name[i-1]), camera_pred[0, i:i+1].cpu(), epoch)

                for i in range(1, lidar_pred.size(1)):
                    self.recorder.tensorboard.add_image(
                        "{}_Pred_cls_{:02d}_{}".format(mode, i-1, self.mapped_cls_name[i-1]), lidar_pred[0, i:i+1].cpu(), epoch)

                # record entropy
                self.recorder.tensorboard.add_image(
                    "{}_PredEntropy".format(mode), pcd_entropy[0].unsqueeze(0), epoch)
                self.recorder.tensorboard.add_image(
                    "{}_RGBPredEntropy".format(mode), img_entropy[0].unsqueeze(0), epoch)
                self.recorder.tensorboard.add_image(
                    "{}_RGBGuideWeight".format(mode), img_guide_weight[0].unsqueeze(0), epoch)
                self.recorder.tensorboard.add_image(
                    "{}_PCDGuideWeight".format(mode), pcd_guide_weight[0].unsqueeze(0), epoch)

                for i in range(1, lidar_pred.size(1)):
                    self.recorder.tensorboard.add_image("{}_Label_cls_{:02d}_{}".format(
                        mode, i-1, self.mapped_cls_name[i-1]), input_label[0:1].eq(i).cpu(), epoch)

                self.recorder.tensorboard.add_image(
                    "{}_RGB".format(mode), img_feature[0].cpu(), epoch)

            log_str = ">>> {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} Recall {:0.4f}".format(
                mode, loss_meter.avg, mean_acc.item(), mean_iou.item(), mean_recall.item())
            self.recorder.logger.info(log_str)

        result_metrics = {
            "Acc": mean_acc.item(),
            "IOU": mean_iou.item(),
            "Recall": mean_recall.item(),
            "last": 0
        }


        return result_metrics