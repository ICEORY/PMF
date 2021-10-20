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


class Inference(object):
    def __init__(self, settings: Option, model: nn.Module, recorder: Recorder):
        self.settings = settings
        self.model = model.cuda()
        self.recorder = recorder
        self.knn_flag = settings.config["post"]["KNN"]["use"]

        self.knn_post = pc_processor.postproc.KNN(
            params=settings.config["post"]["KNN"]["params"],
            nclasses=self.settings.n_classes)

        self.val_loader, self.salsa_loader = self._initDataloader()
        self.prediction_path = os.path.join(self.settings.save_path, "preds")

        self.evaluator = pc_processor.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device("cpu"), ignore=[0])
        self.pixel_eval = pc_processor.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device("cpu"), ignore=[0])
        if self.knn_flag:
            self.recorder.logger.info("using KNN Post Process")

    def _initDataloader(self):
        if self.settings.dataset == "SemanticKitti":
            valset = pc_processor.dataset.semantic_kitti.SemanticKitti(
                root=self.settings.data_root,
                sequences=[8],
                config_path="../../pc_processor/dataset/semantic_kitti/semantic-kitti.yaml",
                has_label=self.settings.has_label,
                has_image=True
            )
        else:
            raise ValueError(
                "invalid dataset: {}".format(self.settings.dataset))

        val_perspective_loader = pc_processor.dataset.PerspectiveViewLoader(
            dataset=valset,
            config=self.settings.config,
            is_train=False,
            return_uproj=True)

        val_loader = torch.utils.data.DataLoader(
            val_perspective_loader,
            batch_size=1,
            num_workers=self.settings.n_threads,
            shuffle=False,
            drop_last=False
        )

        return val_loader, val_perspective_loader

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
            for i, (input_feature, input_mask, input_label, uproj_x_idx, uproj_y_idx, uproj_depth) in enumerate(self.val_loader):
                t_process_start = time.time()
                uproj_x_idx = uproj_x_idx[0].long().cuda()
                uproj_y_idx = uproj_y_idx[0].long().cuda()
                uproj_depth = uproj_depth[0].cuda()

                input_feature = input_feature.cuda()
                proj_depth = input_feature[0, 0, ...].clone()
                proj_depth = proj_depth - proj_depth.eq(0).float()
                # padding
                h_pad = self.settings.config["sensor"]["h_pad"]
                w_pad = self.settings.config["sensor"]["w_pad"]
                padding_layer = torch.nn.ZeroPad2d(
                    (w_pad, w_pad, h_pad, h_pad))

                input_feature = padding_layer(input_feature)
                input_mask = input_mask.cuda()
                input_mask = padding_layer(input_mask)
                input_feature[:, 0:5] = (
                    input_feature[:, 0:5] - feature_mean) / feature_std * \
                    input_mask.unsqueeze(1).expand_as(input_feature[:, 0:5])
                pcd_feature = input_feature[:, 0:5]
                img_feature = input_feature[:, 5:8]

                input_label = input_label.long().cuda()
                # do post process
                pred_output, _ = self.model(pcd_feature, img_feature)
                                    
                # do crop
                pred_output = pred_output[:, :, h_pad: h_pad +
                                          input_label.size(1), w_pad:w_pad+input_label.size(2)]
                pred_argmax = pred_output[0].argmax(dim=0)
                argmax = pred_output.argmax(dim=1)
                if self.settings.has_label:
                    self.pixel_eval.addBatch(argmax, input_label)
                    iter_miou, _ = self.pixel_eval.getIoU()
                if self.knn_flag:
                    # knn post process
                    unproj_argmax = self.knn_post(
                        proj_depth,
                        uproj_depth,
                        pred_argmax,
                        uproj_y_idx,
                        uproj_x_idx,
                    )
                else:
                    unproj_argmax = pred_argmax[uproj_x_idx, uproj_y_idx]

                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)

                pred_np_origin = self.salsa_loader.dataset.class_map_lut_inv[pred_np]

                if self.settings.has_label:
                    sem_label, _ = self.salsa_loader.dataset.loadLabelByIndex(
                        i)
                    self.evaluator.addBatch(
                        pred_np, self.salsa_loader.dataset.class_map_lut[sem_label])

                    
                seq_id, frame_id = self.salsa_loader.dataset.parsePathInfoByIndex(
                    i)
                pred_path = os.path.join(
                    self.prediction_path, "sequences", seq_id, "predictions")
                if not os.path.isdir(pred_path):
                    os.makedirs(pred_path)
                pred_result_path = os.path.join(
                    pred_path, "{}.label".format(frame_id))
                pred_np_origin.tofile(pred_result_path)

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

                if self.settings.is_debug:
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
                cls_eval_table.add_row([i, self.salsa_loader.dataset.mapped_cls_name[i], iou.item(), cls_acc[i].cpu(
                ).item(), cls_recall[i].cpu().item()])
                latext_str += " & {:0.1f}".format(iou * 100)
        latext_str +=  " & {:0.1f}".format(m_iou.cpu().item() * 100)
        self.recorder.logger.info(cls_eval_table)
        self.recorder.logger.info("---- Latext Format String -----")
        self.recorder.logger.info(latext_str)

        conf_matrix = self.evaluator.conf_matrix.clone().cpu()
        conf_matrix[0] = 0
        conf_matrix[:, 0] = 0
        distribution_table = PrettyTable(["Class Name", "Number of points", "Percentage"])
        dist_data = conf_matrix.sum(0)
        for i in range(self.settings.n_classes):
            distribution_table.add_row([self.salsa_loader.dataset.mapped_cls_name[i], dist_data[i].item(), (dist_data[i]/dist_data.sum()).item()])
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
                self.salsa_loader.dataset.mapped_cls_name[i]))
        acc_table = PrettyTable(table_title)
        for i in range(1, self.settings.n_classes):
            row_data = ["{}".format(
                self.salsa_loader.dataset.mapped_cls_name[i])]
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
                self.salsa_loader.dataset.mapped_cls_name[i]))
        recall_table = PrettyTable(table_title)
        for i in range(1, self.settings.n_classes):
            row_data = ["{}".format(
                self.salsa_loader.dataset.mapped_cls_name[i])]
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
                cls_eval_table.add_row([i, self.salsa_loader.dataset.mapped_cls_name[i], iou.item(), cls_pacc[i].cpu(
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
            distribution_table.add_row([self.salsa_loader.dataset.mapped_cls_name[i], dist_data[i].item()])
        self.recorder.logger.info("---- Data Distribution -----")
        self.recorder.logger.info(distribution_table)


        self.recorder.logger.info("---- confusion matrix original data -----")
        self.recorder.logger.info(conf_matrix)
        # get acc matrics
        acc_data = conf_matrix.float() / (conf_matrix.sum(1, keepdim=True).float() + 1e-8)
        table_title = [" "]
        for i in range(1, self.settings.n_classes):
            table_title.append("{}".format(
                self.salsa_loader.dataset.mapped_cls_name[i]))
        acc_table = PrettyTable(table_title)
        for i in range(1, self.settings.n_classes):
            row_data = ["{}".format(
                self.salsa_loader.dataset.mapped_cls_name[i])]
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
                self.salsa_loader.dataset.mapped_cls_name[i]))
        recall_table = PrettyTable(table_title)
        for i in range(1, self.settings.n_classes):
            row_data = ["{}".format(
                self.salsa_loader.dataset.mapped_cls_name[i])]
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
