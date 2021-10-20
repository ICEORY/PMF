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
import sensat_tools
from torchvision import transforms

class Inference(object):
    def __init__(self, settings: Option, model: nn.Module, recorder: Recorder):
        self.settings = settings
        self.model = model.cuda()
        self.recorder = recorder

        self.valset = self._initDataloader()
        self.prediction_path = os.path.join(self.settings.save_path, "preds")

        self.evaluator = pc_processor.metrics.IOUEval(
            n_classes=self.settings.nclasses, device=torch.device("cpu"), ignore=[0])
        self.pixel_eval = pc_processor.metrics.IOUEval(
            n_classes=self.settings.nclasses, device=torch.device("cpu"), ignore=[0])
        if self.settings.has_label:
            self.data_split = "val"
        else:
            self.data_split = "test"

        self.use_knn = settings.config["post"]["KNN"]["use"]
        if self.use_knn:
            self.recorder.logger.info("use knn")
            self.knn_post = pc_processor.postproc.KNN(
                params=settings.config["post"]["KNN"]["params"],
                nclasses=self.settings.nclasses)

        self.use_tta = settings.config["post"]["tta"]["use"]
        if self.use_tta:
            self.recorder.logger.info("use tta")
            self.lr_flip = transforms.Compose([
                transforms.RandomHorizontalFlip(1),
            ])
            self.updowm_flip = transforms.Compose([
                transforms.RandomVerticalFlip(1)
            ])
            self.pad = transforms.Compose([
                transforms.Pad(padding=16, fill=0, padding_mode='constant')
            ])

    def _initDataloader(self):
        if self.settings.dataset == "SensatUrban":
            if self.settings.has_label:
                split = "val"
            else:
                split = "test"   
            valset = pc_processor.dataset.SensatUrban(
                root_path=self.settings.data_root,
                split=split,
                keep_idx=True,
                use_crop=False
            )
        else:
            raise ValueError(
                "invalid dataset: {}".format(self.settings.dataset))

        return valset

    def run(self):
        self.model.eval()
        self.evaluator.reset()
        self.pixel_eval.reset()

        feature_mean = torch.Tensor(self.settings.feature_mean).unsqueeze(
            0).unsqueeze(2).unsqueeze(2).cuda()
        feature_std = torch.Tensor(self.settings.feature_std).unsqueeze(
            0).unsqueeze(2).unsqueeze(2).cuda()

        with torch.no_grad():
            t_start = time.time()
            data_len = len(self.valset)
            for i in range(data_len):
                t_process_start = time.time()
                data_frame = self.valset.readDataByIndex(i)
                label_map_tensor = torch.from_numpy(data_frame["label_map"]).long()
                h = data_frame["feature_map"].shape[1]
                w = data_frame["feature_map"].shape[2]
                confidence_map = torch.zeros((self.settings.nclasses, h, w)).float()

                for img_size in self.settings.img_size:
                    rows = math.ceil(data_frame["feature_map"].shape[1] / img_size)
                    cols = math.ceil(data_frame["feature_map"].shape[2] / img_size)

                    for r in range(rows):
                        h_start = r * img_size
                        h_end = (r + 1) * img_size
                        if h_end > h:
                            h_end = h
                            h_start = h - img_size
                            if h_start < 0:
                                h_start = 0
                        
                        for c in range(cols):
                            # print(i, r, c)
                            w_start = c * img_size
                            w_end = (c + 1) * img_size
                            if w_end > w:
                                w_end = w
                                w_start = w - img_size
                                if w_start < 0:
                                    w_start = 0
                            crop_feature_map = np.zeros((8, img_size, img_size))
                            crop_feature_map[:, :h_end-h_start, :w_end-w_start] = data_frame["feature_map"][:, h_start:h_end, w_start:w_end]

                            input_feature = torch.from_numpy(
                                crop_feature_map).float().cuda().unsqueeze(0)
                            input_mask = input_feature[:, 4]
                            input_feature = (input_feature - feature_mean)/feature_std * input_mask.unsqueeze(1)
                            input_pcd_feature = input_feature[:, 0:5]
                            input_img_feature = input_feature[:, 5:8]

                            if self.use_tta:
                                # data augmentation: rot90, rot180, vflip, hflip, transform, pad

                                aug_pcd_1 = input_pcd_feature.rot90(1, (2, 3))
                                aug_img_1 = input_img_feature.rot90(1, (2, 3))

                                aug_pcd_2 = input_pcd_feature.rot90(2, (2, 3))
                                aug_img_2 = input_img_feature.rot90(2, (2, 3))

                                aug_pcd_3 = self.lr_flip(input_pcd_feature)
                                aug_img_3 = self.lr_flip(input_img_feature)

                                aug_pcd_4 = self.updowm_flip(input_pcd_feature)
                                aug_img_4 = self.updowm_flip(input_img_feature)

                                aug_pcd_5 = input_pcd_feature.permute(0, 1, 3, 2)
                                aug_img_5 = input_img_feature.permute(0, 1, 3, 2)

                                de_pad = transforms.Compose([transforms.CenterCrop(size=img_size)])
                                aug_pcd_6 = self.pad(input_pcd_feature)
                                aug_img_6 = self.pad(input_img_feature)

                                # forward
                                pred_output_0, _ = self.model(input_pcd_feature, input_img_feature)
                                pred_output_1, _ = self.model(aug_pcd_1, aug_img_1)
                                pred_output_2, _ = self.model(aug_pcd_2, aug_img_2)
                                pred_output_3, _ = self.model(aug_pcd_3, aug_img_3)
                                pred_output_4, _ = self.model(aug_pcd_4, aug_img_4)
                                pred_output_5, _ = self.model(aug_pcd_5, aug_img_5)
                                pred_output_6, _ = self.model(aug_pcd_6, aug_img_6)

                                deau_pred_output_1 = pred_output_1.rot90(3, (2, 3))
                                deau_pred_output_2 = pred_output_2.rot90(2, (2, 3))
                                deau_pred_output_3 = self.lr_flip(pred_output_3)
                                deau_pred_output_4 = self.updowm_flip(pred_output_4)
                                deau_pred_output_5 = pred_output_5.permute(0, 1, 3, 2)
                                deau_pred_output_6 = de_pad(pred_output_6)

                                pred_output = (pred_output_0 + deau_pred_output_1+ deau_pred_output_2
                                +deau_pred_output_3+deau_pred_output_4
                                +deau_pred_output_5+deau_pred_output_6)

                            else:
                                pred_output, _ = self.model(input_pcd_feature, input_img_feature)
                            
                            confidence_map[:, h_start:h_end, w_start:w_end] += pred_output[0].cpu()
                
                argmax = confidence_map.unsqueeze(0).argmax(dim=1)
                if self.settings.has_label:
                    self.pixel_eval.addBatch(argmax, label_map_tensor+1)

                full_pred_map = argmax[0].cpu()
                
                if self.settings.has_label:
                    iter_miou, _ = self.pixel_eval.getIoU()

                h_idx = data_frame["h_idx"]
                w_idx = data_frame["w_idx"]

                if self.use_knn:
                    file_name = self.valset.readFileNameByIndex(i).replace(".bin", ".ply")
                    data = sensat_tools.read_ply(filename=os.path.join(self.valset.split_folder, file_name))
                    # print("debug: z, ", data["z"].shape, data["z"].dtype)
                    uproj_depth = torch.from_numpy(data["z"].copy()).float()
                    print("run knn")
                    pred_tensor = self.knn_post(
                        torch.from_numpy(data_frame["feature_map"][0]).float(),
                        uproj_depth,
                        full_pred_map,
                        torch.from_numpy(w_idx).long(),
                        torch.from_numpy(h_idx).long()
                    )
                    print("knn finish")
                else:
                    pred_tensor = full_pred_map[h_idx, w_idx]

                zero_idx = pred_tensor.eq(0).nonzero(as_tuple=False)
                zero_num = pred_tensor.eq(0).sum()
                if zero_num > 0:
                    print("warning zero_num: ", zero_num, " set zero to ground")
                    pred_tensor[zero_idx] = 1
                pred_np = pred_tensor.cpu().numpy().astype(np.uint8)

                labelfile_name = self.valset.readFileNameByIndex(i)
                
                if self.settings.has_label:
                    label_np = self.valset.readLabelByIndex(i) + 1
                    self.evaluator.addBatch(
                        pred_np, label_np)
                    
                pred_path = os.path.join(
                    self.prediction_path, "{}_preds".format(self.data_split))


                if not os.path.isdir(pred_path):
                    os.makedirs(pred_path)
                pred_result_path = os.path.join(
                    pred_path, labelfile_name.replace(".bin", ".label"))
                (pred_np-1).tofile(pred_result_path)

                # save score
                numpy_score = confidence_map.unsqueeze(0).numpy().astype(np.float32)
                score_path = os.path.join(
                    self.prediction_path, "{}_scors".format(self.data_split)
                )
                if not os.path.isdir(score_path):
                    os.makedirs(score_path)
                np.save(os.path.join(score_path, labelfile_name.strip(".bin")), numpy_score)

                t_proces_end = time.time()
                process_time = t_proces_end-t_process_start
                data_time = t_process_start - t_start
                t_start = time.time()
                log_str = "Iter [{:04d}|{:04d}] Datatime: {:0.3f} ProcessTime: {:0.3f}".format(
                    i, data_len, data_time, process_time)
                if self.settings.has_label:
                    log_str += " meanIOU {:0.4f}".format(
                        iter_miou.cpu().item())
                self.recorder.logger.info(log_str)

                if self.settings.is_debug:
                    break

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
                cls_eval_table.add_row([i, self.valset.mapped_cls_name[i-1], iou.item(), cls_acc[i].cpu(
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
        for i in range(self.settings.nclasses):
            distribution_table.add_row([self.valset.mapped_cls_name[i-1], dist_data[i].item(
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
        for i in range(1, self.settings.nclasses):
            table_title.append("{}".format(
                self.valset.mapped_cls_name[i-1]))
        acc_table = PrettyTable(table_title)
        for i in range(1, self.settings.nclasses):
            row_data = ["{}".format(
                self.valset.mapped_cls_name[i-1])]
            for j in range(1, self.settings.nclasses):
                row_data.append("{:0.1f}".format(acc_data[i, j]*100))
            acc_table.add_row(row_data)
        self.recorder.logger.info("---- ACC matrix ----------------")
        self.recorder.logger.info(acc_table)

        # get recall matrics
        recall_data = conf_matrix.float() / (conf_matrix.sum(0, keepdim=True).float()+1e-8)
        table_title = [" "]
        for i in range(1, self.settings.nclasses):
            table_title.append("{}".format(
                self.valset.mapped_cls_name[i-1]))
        recall_table = PrettyTable(table_title)
        for i in range(1, self.settings.nclasses):
            row_data = ["{}".format(
                self.valset.mapped_cls_name[i-1])]
            for j in range(1, self.settings.nclasses):
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
                cls_eval_table.add_row([i, self.valset.mapped_cls_name[i-1], iou.item(), cls_pacc[i].cpu(
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
        for i in range(self.settings.nclasses):
            distribution_table.add_row(
                [self.valset.mapped_cls_name[i-1], dist_data[i].item()])
        self.recorder.logger.info("---- Data Distribution -----")
        self.recorder.logger.info(distribution_table)

        self.recorder.logger.info("---- confusion matrix original data -----")
        self.recorder.logger.info(conf_matrix)
        # get acc matrics
        acc_data = conf_matrix.float() / (conf_matrix.sum(1, keepdim=True).float() + 1e-8)
        table_title = [" "]
        for i in range(1, self.settings.nclasses):
            table_title.append("{}".format(
                self.valset.mapped_cls_name[i-1]))
        acc_table = PrettyTable(table_title)
        for i in range(1, self.settings.nclasses):
            row_data = ["{}".format(
                self.valset.mapped_cls_name[i-1])]
            for j in range(1, self.settings.nclasses):
                row_data.append("{:0.1f}".format(acc_data[i, j]*100))
            acc_table.add_row(row_data)
        self.recorder.logger.info("---- ACC matrix ----------------")
        self.recorder.logger.info(acc_table)

        # get recall matrics
        recall_data = conf_matrix.float() / (conf_matrix.sum(0, keepdim=True).float()+1e-8)
        table_title = [" "]
        for i in range(1, self.settings.nclasses):
            table_title.append("{}".format(
                self.valset.mapped_cls_name[i-1]))
        recall_table = PrettyTable(table_title)
        for i in range(1, self.settings.nclasses):
            row_data = ["{}".format(
                self.valset.mapped_cls_name[i-1])]
            for j in range(1, self.settings.nclasses):
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
            nclasses=self.settings.nclasses,
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
