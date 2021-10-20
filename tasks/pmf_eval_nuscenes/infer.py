# inference and save result to file, report pointwise result

import torch
from option import Option
import argparse
import torch.nn as nn
import datetime
import pc_processor
from pc_processor.checkpoint import Recorder
import numpy as np
import os
import time
import math
from prettytable import PrettyTable
from nus_perspective_loader import NusPerspectiveViewLoader


def getMergePred(point_idx_list, pred_conf_list, pred_argmax_list, pc_size):
    # merge result
    merge_conf = torch.zeros((6, pc_size)).cuda()
    merge_argmax = torch.zeros((6, pc_size)).cuda() - 1
    merge_argmax = merge_argmax.long()
    for j, pred_conf in enumerate(pred_conf_list):
        merge_conf[j, point_idx_list[j]] = pred_conf
    for j, pred_argmax in enumerate(pred_argmax_list):
        merge_argmax[j, point_idx_list[j]] = pred_argmax
    argmax = merge_conf.argmax(dim=0)
    merge_pred = torch.zeros(pc_size).cuda().long() - 1
    for j in range(pc_size):
        merge_pred[j] = merge_argmax[argmax[j], j]

    # check results
    zero_num = merge_pred.eq(0).sum()
    zero_idx = merge_pred.eq(0).nonzero(as_tuple=False)
    merge_pred[zero_idx] = 0
    if zero_num > 0:
        print("warning zero_num: ", zero_num, " set zero to undefined")
    return merge_pred


class Inference(object):
    def __init__(self, settings: Option, model: nn.Module, recorder: Recorder):
        self.settings = settings
        self.model = model.cuda()
        self.recorder = recorder
        self.knn_flag = settings.config["post"]["KNN"]["use"]

        self.knn_post = pc_processor.postproc.KNN(
            params=settings.config["post"]["KNN"]["params"],
            nclasses=self.settings.n_classes)

        self.val_loader, self.nus_loader = self._initDataloader()
        self.prediction_path = os.path.join(self.settings.save_path, "preds")

        self.evaluator = pc_processor.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device("cpu"), ignore=[0])
        self.pixel_eval = pc_processor.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device("cpu"), ignore=[0])
        if self.knn_flag:
            self.recorder.logger.info("using KNN Post Process")
        if self.settings.has_label:
            self.data_split = "val"
        else:
            self.data_split = "test"

    def _initDataloader(self):
        if self.settings.dataset == "nuScenes":
            if self.settings.is_debug:
                version = "v1.0-mini"
                split = "val"
            else:
                if self.settings.has_label:
                    version = "v1.0-trainval"
                    split = "val"
                else:
                    version = "v1.0-test"   
                    split = "test"   
            valset = pc_processor.dataset.nuScenes.Nuscenes(
                root=self.settings.data_root, version=version, split=split,
            )
        else:
            raise ValueError(
                "invalid dataset: {}".format(self.settings.dataset))

        val_nus_loader = NusPerspectiveViewLoader(
            dataset=valset,
            config=self.settings.config)

        val_loader = torch.utils.data.DataLoader(
            val_nus_loader,
            batch_size=1,
            num_workers=self.settings.n_threads,
            shuffle=False,
            drop_last=False
        )

        return val_loader, val_nus_loader

    def run(self):
        self.model.eval()
        self.evaluator.reset()
        self.pixel_eval.reset()
        with torch.no_grad():
            t_start = time.time()
            feature_mean = torch.Tensor(self.settings.config["sensor"]["img_mean"]).unsqueeze(
                0).unsqueeze(2).unsqueeze(2).cuda()
            feature_std = torch.Tensor(self.settings.config["sensor"]["img_stds"]).unsqueeze(
                0).unsqueeze(2).unsqueeze(2).cuda()

            cam_count = 0
            point_idx_list = []
            pred_conf_list = []
            pred_argmax_list = []
            for i, (input_feature, input_mask, input_label, uproj_x_idx, uproj_y_idx, uproj_depth, point_idx, point_size) in enumerate(self.val_loader):
                t_process_start = time.time()
                uproj_x_idx = uproj_x_idx[0].long().cuda()
                uproj_y_idx = uproj_y_idx[0].long().cuda()
                uproj_depth = uproj_depth[0].cuda()
                point_idx = point_idx[0].long().cuda()

                input_feature = input_feature.cuda()
                proj_depth = input_feature[0, 0, ...].clone()
                proj_depth = proj_depth - proj_depth.eq(0).float()
                # crop
                img_h = self.settings.config["sensor"]["proj_h"]
                h_pad = input_feature.size(2) - img_h

                input_feature = input_feature[:, :, h_pad:, :]
                input_mask = input_mask.cuda()
                input_mask = input_mask[:, h_pad:, :]
                input_feature[:, 0:5] = (
                    input_feature[:, 0:5] - feature_mean) / feature_std * \
                    input_mask.unsqueeze(1).expand_as(input_feature[:, 0:5])
                pcd_feature = input_feature[:, 0:5]
                img_feature = input_feature[:, 5:8]

                input_label = input_label.long().cuda()
                # do post process
                pred_output, _ = self.model(pcd_feature, img_feature)

                # do padding
                padding_layer = torch.nn.ZeroPad2d((0, 0, h_pad, 0))
                pred_output = padding_layer(pred_output)

                pred_conf, pred_argmax = pred_output[0].max(dim=0)
                argmax = pred_output.argmax(dim=1)
                if self.settings.has_label:
                    self.pixel_eval.addBatch(argmax, input_label)
                    iter_miou, _ = self.pixel_eval.getIoU()

                if self.knn_flag:
                    unproj_argmax = self.knn_post(
                        proj_depth,
                        uproj_depth,
                        pred_argmax,
                        uproj_y_idx,
                        uproj_x_idx,
                    )
                else:
                    unproj_argmax = pred_argmax[uproj_x_idx, uproj_y_idx]

                # merge 6 camera predictions
                cam_count += 1
                point_idx_list.append(point_idx)
                unproj_conf = pred_conf[uproj_x_idx, uproj_y_idx]
                pred_conf_list.append(unproj_conf)
                pred_argmax_list.append(unproj_argmax)

                if cam_count == 6:
                    # check lidar token
                    token = lidar_token = self.nus_loader.dataset.token_list[i]["lidar_token"]
                    for j in range(i-5, i):
                        lidar_token = self.nus_loader.dataset.token_list[j]["lidar_token"]
                        assert token == lidar_token

                    # merge predictions
                    merge_pred = getMergePred(
                        point_idx_list, pred_conf_list, pred_argmax_list, point_size[0].long().item())
 

                    valid_pred = merge_pred.ne(-1).long()
                    merge_pred = valid_pred * merge_pred
                    pred_np = merge_pred.cpu().numpy()
                    pred_np = pred_np.reshape((-1)).astype(np.int32)

                    # reset cache data
                    cam_count = 0
                    point_idx_list = []
                    pred_conf_list = []
                    pred_argmax_list = []

                    # save to file
                    if self.settings.has_label:
                        _, sem_label_str, _ = self.nus_loader.dataset.loadDataByIndex(
                            i)
                        sem_label = self.nus_loader.dataset.labelMapping(
                            sem_label_str)
                        sem_label = sem_label * valid_pred.cpu().numpy()
                        self.evaluator.addBatch(
                            pred_np, sem_label)

                    pred_path = os.path.join(
                        self.prediction_path, "lidarseg", self.data_split)

                    lidar_token = self.nus_loader.dataset.token_list[i]["lidar_token"]
                    if not os.path.isdir(pred_path):
                        os.makedirs(pred_path)
                    pred_result_path = os.path.join(
                        pred_path, "{}_lidarseg.bin".format(lidar_token))
                    pred_np.tofile(pred_result_path)

                #
                t_proces_end = time.time()
                process_time = t_proces_end-t_process_start
                data_time = t_process_start - t_start
                t_start = time.time()
                log_str = "Iter [{:04d}|{:04d}] Datatime: {:0.3f} ProcessTime: {:0.3f}".format(
                    i, len(self.val_loader), data_time, process_time)
                if self.settings.has_label:
                    log_str += " meanIOU {:0.4f}".format(
                        iter_miou.cpu().item())
                print(log_str)

                if self.settings.is_debug and i > 10:
                    break

        #########################################################################################
        #########################################################################################
        if not self.settings.has_label:
            return
        # show results
        # get result
        m_acc, cls_acc = self.evaluator.getAcc()
        m_recall, cls_recall = self.evaluator.getRecall()
        m_iou, cls_iou = self.evaluator.getIoU()

        self.recorder.logger.info(
            "============== Point-wise Evaluation Results (3D eval) ===================")
        log_str = "Acc avg: {:.4f}, IOU avg: {:.4f}, Recall avg: {:.4f}".format(
            m_acc.item(), m_iou.item(), m_recall.item())
        self.recorder.logger.info(log_str)
        cls_eval_table = PrettyTable(
            ["ClassIdx", "class_name", "IOU", "Acc", "Recall"])
        latext_str = ""  # " {:0.1f}".format(m_iou.cpu().item() * 100)
        for i, iou in enumerate(cls_iou.cpu()):
            if i not in [0]:
                cls_eval_table.add_row([i, self.nus_loader.dataset.mapped_cls_name[i], iou.item(), cls_acc[i].cpu(
                ).item(), cls_recall[i].cpu().item()])
                latext_str += " & {:0.1f}".format(iou * 100)
        latext_str += " & {:0.1f}".format(m_iou.cpu().item() * 100)
        self.recorder.logger.info(cls_eval_table)
        self.recorder.logger.info("---- Latext Format String -----")
        self.recorder.logger.info(latext_str)

        conf_matrix = self.evaluator.conf_matrix.clone().cpu()
        conf_matrix[0] = 0
        conf_matrix[:, 0] = 0
        distribution_table = PrettyTable(
            ["Class Name", "Number of points", "Percentage"])
        dist_data = conf_matrix.sum(0)
        for i in range(self.settings.n_classes):
            distribution_table.add_row([self.nus_loader.dataset.mapped_cls_name[i], dist_data[i].item(
            ), (dist_data[i]/dist_data.sum()).item()])
        self.recorder.logger.info("---- Data Distribution -----")
        self.recorder.logger.info(distribution_table)
        # compute fwIoU
        freqw = dist_data[1:] / dist_data[1:].sum()
        freq_iou = (cls_iou[1:] * freqw).sum()
        self.recorder.logger.info("fwIoU: {}".format(freq_iou.item()))

        self.recorder.logger.info("---- confusion matrix original data -----")
        self.recorder.logger.info(conf_matrix)
        # get acc matrics
        acc_data = conf_matrix.float() / (conf_matrix.sum(1, keepdim=True).float() + 1e-8)
        table_title = [" "]
        for i in range(1, self.settings.n_classes):
            table_title.append("{}".format(
                self.nus_loader.dataset.mapped_cls_name[i]))
        acc_table = PrettyTable(table_title)
        for i in range(1, self.settings.n_classes):
            row_data = ["{}".format(
                self.nus_loader.dataset.mapped_cls_name[i])]
            for j in range(1, self.settings.n_classes):
                row_data.append("{:0.1f}".format(acc_data[i, j]*100))
            acc_table.add_row(row_data)
        self.recorder.logger.info("---- ACC matrix ----------------")
        self.recorder.logger.info(acc_table)

        # get recall matrics
        recall_data = conf_matrix.float() / (conf_matrix.sum(0, keepdim=True).float()+1e-8)
        table_title = [" "]
        for i in range(1, self.settings.n_classes):
            table_title.append("{}".format(
                self.nus_loader.dataset.mapped_cls_name[i]))
        recall_table = PrettyTable(table_title)
        for i in range(1, self.settings.n_classes):
            row_data = ["{}".format(
                self.nus_loader.dataset.mapped_cls_name[i])]
            for j in range(1, self.settings.n_classes):
                row_data.append("{:0.1f}".format(recall_data[i, j]*100))
            recall_table.add_row(row_data)
        self.recorder.logger.info("---- Recall matrix ----------------")
        self.recorder.logger.info(recall_table)

        self.recorder.logger.info(
            "============== Pixel-wise Evaluation Results (2D eval) ===================")
        m_pacc, cls_pacc = self.pixel_eval.getAcc()
        m_precall, cls_precall = self.pixel_eval.getRecall()
        m_piou, cls_piou = self.pixel_eval.getIoU()

        log_str = "Pixel Acc avg: {:.4f}, IOU avg: {:.4f}, Recall avg: {:.4f}".format(
            m_pacc.item(), m_piou.item(), m_precall.item())
        self.recorder.logger.info(log_str)

        cls_eval_table = PrettyTable(
            ["ClassIdx", "class_name", "IOU", "Acc", "Recall"])
        latext_str = ""  # " {:0.1f}".format(m_piou.cpu().item() * 100)
        for i, iou in enumerate(cls_piou.cpu()):
            if i not in [0]:
                cls_eval_table.add_row([i, self.nus_loader.dataset.mapped_cls_name[i], iou.item(), cls_pacc[i].cpu(
                ).item(), cls_precall[i].cpu().item()])
                latext_str += " & {:0.1f}".format(iou * 100)
        latext_str += " & {:0.1f}".format(m_piou.cpu().item() * 100)
        self.recorder.logger.info(cls_eval_table)
        self.recorder.logger.info("---- Latext Format String -----")
        self.recorder.logger.info(latext_str)

        conf_matrix = self.pixel_eval.conf_matrix.clone().cpu()
        conf_matrix[0] = 0
        conf_matrix[:, 0] = 0
        distribution_table = PrettyTable(["Class Name", "Number of points"])
        dist_data = conf_matrix.sum(0)
        for i in range(self.settings.n_classes):
            distribution_table.add_row(
                [self.nus_loader.dataset.mapped_cls_name[i], dist_data[i].item()])
        self.recorder.logger.info("---- Data Distribution -----")
        self.recorder.logger.info(distribution_table)

        self.recorder.logger.info("---- confusion matrix original data -----")
        self.recorder.logger.info(conf_matrix)
        # get acc matrics
        acc_data = conf_matrix.float() / (conf_matrix.sum(1, keepdim=True).float() + 1e-8)
        table_title = [" "]
        for i in range(1, self.settings.n_classes):
            table_title.append("{}".format(
                self.nus_loader.dataset.mapped_cls_name[i]))
        acc_table = PrettyTable(table_title)
        for i in range(1, self.settings.n_classes):
            row_data = ["{}".format(
                self.nus_loader.dataset.mapped_cls_name[i])]
            for j in range(1, self.settings.n_classes):
                row_data.append("{:0.1f}".format(acc_data[i, j]*100))
            acc_table.add_row(row_data)
        self.recorder.logger.info("---- ACC matrix ----------------")
        self.recorder.logger.info(acc_table)

        # get recall matrics
        recall_data = conf_matrix.float() / (conf_matrix.sum(0, keepdim=True).float()+1e-8)
        table_title = [" "]
        for i in range(1, self.settings.n_classes):
            table_title.append("{}".format(
                self.nus_loader.dataset.mapped_cls_name[i]))
        recall_table = PrettyTable(table_title)
        for i in range(1, self.settings.n_classes):
            row_data = ["{}".format(
                self.nus_loader.dataset.mapped_cls_name[i])]
            for j in range(1, self.settings.n_classes):
                row_data.append("{:0.1f}".format(recall_data[i, j]*100))
            recall_table.add_row(row_data)
        self.recorder.logger.info("---- Recall matrix ----------------")
        self.recorder.logger.info(recall_table)


class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings
        # init gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = self.settings.gpu
        self.settings.check_path()
        # set random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        self.recorder = pc_processor.checkpoint.Recorder(
            self.settings, self.settings.save_path, use_tensorboard=False)
        # init model
        self.model = self._initModel()

        # load checkpoint
        self._loadCheckpoint()

        # init inference
        self.inference = Inference(self.settings, self.model, self.recorder)

    def _initModel(self):
        model = pc_processor.models.PMFNet(
            pcd_channels=5,
            img_channels=3,
            nclasses=self.settings.n_classes,
            base_channels=self.settings.base_channels,
            image_backbone=self.settings.img_backbone,
            imagenet_pretrained=self.settings.imagenet_pretrained
        )
        return model

    def _loadCheckpoint(self):
        if self.settings.pretrained_model is not None:
            if not os.path.isfile(self.settings.pretrained_model):
                raise FileNotFoundError("pretrained model not found: {}".format(
                    self.settings.pretrained_model))
            state_dict = torch.load(
                self.settings.pretrained_model, map_location="cpu")
            self.model.load_state_dict(state_dict)
            if self.recorder is not None:
                self.recorder.logger.info(
                    "loading pretrained weight from: {}".format(self.settings.pretrained_model))
    def run(self):
        t_start = time.time()
        self.inference.run()
        cost_time = time.time() - t_start
        self.recorder.logger.info("==== total cost time: {}".format(
            datetime.timedelta(seconds=cost_time)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Options")
    parser.add_argument("config_path", type=str, metavar="config_path",
                        help="path of config file, type: string")
    parser.add_argument("--id", type=int, metavar="experiment_id", required=False,
                        help="id of experiment", default=0)
    args = parser.parse_args()
    exp = Experiment(Option(args.config_path))
    print("===init env success===")
    exp.run()
