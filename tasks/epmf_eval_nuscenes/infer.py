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
from prettytable import PrettyTable
from tqdm import tqdm
import math

class Inference(object):
    def __init__(self, settings: Option, model: nn.Module, recorder: Recorder):
        self.settings = settings
        self.model = model.cuda()
        self.recorder = recorder
        self.knn_flag = settings.config["post"]["KNN"]["use"]

        self.val_loader, self.nus_loader = self._initDataloader()
        self.prediction_path = os.path.join(self.settings.save_path, "preds")

        self.evaluator = pc_processor.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device("cpu"), ignore=[0])
        self.pixel_eval = pc_processor.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device("cpu"), ignore=[0])
        if self.knn_flag:
            self.knn_post = pc_processor.postproc.KNN(
                params=settings.config["post"]["KNN"]["params"],
                nclasses=self.settings.n_classes)
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
            valset = pc_processor.dataset.nuScenes.NuscenesV2(
                root=self.settings.data_root, version=version, split=split,
            )
        else:
            raise ValueError(
                "invalid dataset: {}".format(self.settings.dataset))

        val_nus_loader = pc_processor.dataset.PerspectiveViewLoaderV2(
            dataset=valset,
            config=self.settings.config,
            is_train=False,
            return_uproj=True)

        val_loader = torch.utils.data.DataLoader(
            val_nus_loader,
            batch_size=1,
            num_workers=self.settings.n_threads,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )

        return val_loader, val_nus_loader

    def run(self):
        self.model.eval()
        self.evaluator.reset()
        self.pixel_eval.reset()
        with torch.no_grad():
            t_start = time.time()
            feature_mean = torch.Tensor(self.settings.config["PVconfig"]["pcd_mean"]).unsqueeze(
                0).unsqueeze(2).unsqueeze(2).cuda()
            feature_std = torch.Tensor(self.settings.config["PVconfig"]["pcd_stds"]).unsqueeze(
                0).unsqueeze(2).unsqueeze(2).cuda()

            cam_count = 0
            pred_conf_full = None
            pred_argmax_full = None
            previous_lidar_token = None
            pbar = tqdm(total=len(self.val_loader))
            for i, (proj_data, xy_index, depth, keep_mask, pointcloud) in enumerate(self.val_loader):
                t_process_start = time.time()

                pc_size = keep_mask.size(1)
                keep_mask_np = keep_mask[0].numpy().astype(np.bool_)
                if pred_conf_full is None:
                    pred_conf_full = np.zeros((pc_size)).astype(np.float32)
                    pred_argmax_full = np.zeros((pc_size)).astype(np.int32)

                uproj_x_idx = xy_index[0, :, 0].long().cuda()
                uproj_y_idx = xy_index[0, :, 1].long().cuda()
                x_min = uproj_x_idx.min()
                y_min = uproj_y_idx.min()
                uproj_x_idx = uproj_x_idx - x_min
                uproj_y_idx = uproj_y_idx - y_min

                uproj_depth = depth[0].cuda()

                input_feature = proj_data[:, :8].cuda()
                proj_depth = input_feature[0, 0, ...].clone()
                proj_depth = proj_depth - proj_depth.eq(0).float()
                # padding
                h_pad = math.ceil(input_feature.size(2) / 64.0) * 64 - input_feature.size(2)
                w_pad = math.ceil(input_feature.size(3) / 64.0) * 64 - input_feature.size(3)
                padding_layer = torch.nn.ZeroPad2d(
                    (w_pad//2, w_pad-w_pad//2, 0, h_pad))

                input_feature = padding_layer(input_feature)
                input_mask = proj_data[:, 8].cuda()
                input_mask = padding_layer(input_mask)
                input_feature[:, 0:5] = (
                    input_feature[:, 0:5] - feature_mean) / feature_std * \
                    input_mask.unsqueeze(1).expand_as(input_feature[:, 0:5])
                pcd_feature = input_feature[:, 0:5]
                img_feature = input_feature[:, 5:8]

                input_label = proj_data[0, 9:10].long().cuda()

                # forward
                if "PMF" in self.settings.net_type:
                    pred_output, _ = self.model(pcd_feature, img_feature)
                else:
                    pred_output = self.model(pcd_feature)
                # do crop
                pred_output = pred_output[:, :, :input_label.size(1), 
                                        w_pad//2:w_pad//2+input_label.size(2)]

                pred_conf, pred_argmax = pred_output[0].max(dim=0)
                argmax = pred_output.argmax(dim=1)
                if self.settings.has_label:
                    self.pixel_eval.addBatch(argmax, input_label)
                    # iter_miou, _ = self.pixel_eval.getIoU() # pixel-wise evaluation
                    iter_miou, _ = self.evaluator.getIoU() # point-wise evaluation

                if self.knn_flag:
                    unproj_argmax = self.knn_post(
                        proj_depth,
                        uproj_depth,
                        pred_argmax,
                        uproj_y_idx,
                        uproj_x_idx,
                    )
                    unproj_conf = self.knn_post(
                        proj_depth,
                        uproj_depth,
                        pred_conf,
                        uproj_y_idx,
                        uproj_x_idx,
                    )
                else:
                    unproj_argmax = pred_argmax[uproj_x_idx, uproj_y_idx]
                    unproj_conf = pred_conf[uproj_x_idx, uproj_y_idx]

                unproj_argmax_np = unproj_argmax.cpu().numpy()
                unproj_conf_np = unproj_conf.cpu().numpy()
                # merge 6 camera predictions
                cam_count += 1
                keep_conf_mask = pred_conf_full[keep_mask_np] < unproj_conf_np
                keep_mask_np[keep_mask_np] = np.logical_and(keep_mask_np[keep_mask_np], keep_conf_mask)
                pred_conf_full[keep_mask_np] = unproj_conf_np[keep_conf_mask]
                pred_argmax_full[keep_mask_np] = unproj_argmax_np[keep_conf_mask]

                # check lidar token
                current_lidar_token = self.nus_loader.dataset.token_list[i]["lidar_token"]
                if previous_lidar_token is None:
                    previous_lidar_token = current_lidar_token
                assert current_lidar_token == previous_lidar_token

                if cam_count == 6:
                    pred_full_tensor = torch.from_numpy(pred_argmax_full).cuda()
                    pred_full_tensor = pred_full_tensor.cpu()

                    """
                    although we adjust the fov of point clouds w.r.t. each image, 
                    there are still very few point clouds without predictions, 
                    e.g., point clouds with distance < 0.5,
                    so we only evaluate the results on the valid points.
                    for testing, one can set these points to class-4 (ego car); 
                    """
                    pred_valid_mask = pred_full_tensor.ne(0)

                    pred_argmax_full = pred_full_tensor.cpu().numpy()
                    pred_np = pred_argmax_full.astype(np.int32)
                    # save to file
                    if self.settings.has_label:
                        _, sem_label_str, _ = self.nus_loader.dataset.loadDataByIndex(i)
                        sem_label = self.nus_loader.dataset.labelMapping(
                            sem_label_str) * pred_valid_mask.cpu().numpy()

                        self.evaluator.addBatch(pred_np, sem_label)

                    if self.settings.save_pred_results:
                        pred_path = os.path.join(
                            self.prediction_path, "lidarseg", self.data_split)

                        lidar_token = self.nus_loader.dataset.token_list[i]["lidar_token"]
                        if not os.path.isdir(pred_path):
                            os.makedirs(pred_path)
                        pred_result_path = os.path.join(
                            pred_path, "{}_lidarseg.bin".format(lidar_token))
                        pred_np = pred_np.astype(np.uint8)
                        pred_np.tofile(pred_result_path)

                    # reset cache data
                    cam_count = 0
                    pred_conf_full = None
                    pred_argmax_full = None
                    previous_lidar_token = None
                #
                t_proces_end = time.time()
                process_time = t_proces_end-t_process_start
                data_time = t_process_start - t_start
                t_start = time.time()
                if self.settings.has_label:
                    pbar.set_postfix({
                        "Datatime": "{:0.3f}".format(data_time),
                        "ProcessTime" : "{:0.3f}".format(process_time),
                        "meanIoU": "{:0.4f}".format(iter_miou.cpu().item())
                    })
                else:
                    pbar.set_postfix({
                        "Datatime": "{:0.3f}".format(data_time),
                        "ProcessTime" : "{:0.3f}".format(process_time)
                    })
                pbar.update(1)

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
        latext_str = ""
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
        latext_str = ""
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
        if self.settings.net_type == "EPMFNet":
            model = pc_processor.models.EPMFNet(
                pcd_channels=5,
                img_channels=3,
                nclasses=self.settings.n_classes,
                base_channels=self.settings.base_channels,
                image_backbone=self.settings.img_backbone,
                imagenet_pretrained=self.settings.imagenet_pretrained
            )
        else:
            raise NotImplementedError(self.settings.net_type)

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
