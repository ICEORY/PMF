# inference and save result to file, report pointwise result

import torch
from option import Option
import argparse
import datetime
import pc_processor
from pc_processor.checkpoint import Recorder
import numpy as np
import os
import time
import math
from prettytable import PrettyTable
from matplotlib import pyplot as plt
import json

class MergePred(object):
    def __init__(self, settings: Option, recorder: Recorder):
        self.settings = settings
        self.recorder = recorder
        self.nus_loader = self._initDataloader()
        self.prediction_path = os.path.join(self.settings.save_path, "preds")

        self.evaluator = pc_processor.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device("cpu"), ignore=[0])
        if self.settings.has_label:
            self.data_split = "val"
        else:
            self.data_split = "test"

        self.submission_json = {
            "meta": {
                "use_camera":  True,
                "use_lidar":  True,
                "use_radar":  False,
                "use_map":   False,
                "use_external": False
            },
        }

    def _initDataloader(self):
        if self.settings.dataset == "NuScenes":
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
                root=self.settings.data_root, version=version, split=split, has_image=False,
            )
        else:
            raise ValueError(
                "invalid dataset: {}".format(self.settings.dataset))

        return valset

    def _mergeResult(self, main_pred, sub_pred):
        invalid_mask = main_pred == 0
        valid_mask = 1 - invalid_mask
        pred = main_pred * valid_mask + sub_pred * invalid_mask
        invalid_mask = pred == 0
        valid_mask = 1 - invalid_mask
        pred = pred * valid_mask + 11 * invalid_mask
        return pred

    def run(self):
        self.evaluator.reset()

        t_start = time.time()
        data_len = len(self.nus_loader)
        for i in range(data_len):
            # print(i)
            t_process_start = time.time()
            lidar_token = self.nus_loader.token_list[i]

            # load main pred result
            main_pred_file = os.path.join(self.settings.main_pred_folder,
                                          "preds/lidarseg/{}/{}_lidarseg.bin".format(self.data_split, lidar_token))
            main_pred = np.fromfile(main_pred_file, dtype=np.int32)

            # load sub pred result
            sub_pred_file = os.path.join(self.settings.sub_pred_folder,
                                         "preds/lidarseg/{}/{}_lidarseg.bin".format(self.data_split, lidar_token))
            sub_pred = np.fromfile(sub_pred_file, dtype=np.int32)

            pred = self._mergeResult(main_pred, sub_pred)
            if self.settings.has_label:
                # load label
                sem_label_str = self.nus_loader.loadLabelByIndex(i)
                sem_label = self.nus_loader.labelMapping(
                    sem_label_str)
                # compute iou
                self.evaluator.addBatch(pred, sem_label)
                iter_miou, _ = self.evaluator.getIoU()

            pred_path = os.path.join(
                self.prediction_path, "lidarseg", self.data_split)

            if not os.path.isdir(pred_path):
                os.makedirs(pred_path)
            pred_result_path = os.path.join(
                pred_path, "{}_lidarseg.bin".format(lidar_token))
            pred = pred.astype(np.uint8)
            pred.tofile(pred_result_path)

            t_proces_end = time.time()
            process_time = t_proces_end-t_process_start
            data_time = t_process_start - t_start
            t_start = time.time()
            log_str = "Iter [{:04d}|{:04d}] Datatime: {:0.3f} ProcessTime: {:0.3f}".format(
                i, len(self.nus_loader), data_time, process_time)
            if self.settings.has_label:
                log_str += " meanIOU {:0.4f}".format(
                    iter_miou.cpu().item())
            print(log_str)

            if self.settings.is_debug and i > 10:
                break
        
        submission_json_path = os.path.join(self.prediction_path, self.data_split)
        if not os.path.isdir(submission_json_path):
            os.makedirs(submission_json_path)
        with open(os.path.join(submission_json_path, "submission.json"), "w") as f:
            json.dump(self.submission_json, f, ensure_ascii=False, indent=4)

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
                cls_eval_table.add_row([i, self.nus_loader.mapped_cls_name[i], iou.item(), cls_acc[i].cpu(
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
            distribution_table.add_row([self.nus_loader.mapped_cls_name[i], dist_data[i].item(
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
                self.nus_loader.mapped_cls_name[i]))
        acc_table = PrettyTable(table_title)
        for i in range(1, self.settings.n_classes):
            row_data = ["{}".format(
                self.nus_loader.mapped_cls_name[i])]
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
                self.nus_loader.mapped_cls_name[i]))
        recall_table = PrettyTable(table_title)
        for i in range(1, self.settings.n_classes):
            row_data = ["{}".format(
                self.nus_loader.mapped_cls_name[i])]
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
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        self.recorder = pc_processor.checkpoint.Recorder(
            self.settings, self.settings.save_path, use_tensorboard=False)

        # init inference
        self.merge_pred = MergePred(self.settings, self.recorder)

    def run(self):
        t_start = time.time()
        self.merge_pred.run()
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
