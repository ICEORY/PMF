import torch
import os
import argparse
import time
import datetime
from option import Option
import torch.nn as nn
import numpy as np
import pc_processor
from pc_processor.checkpoint import Recorder
from prettytable import PrettyTable


class Inference(object):
    def __init__(self, setting: Option, model: nn.Module, recorder: Recorder):
        self.setting = setting
        self.model = model.cuda()
        self.recorder = recorder

        self.val_set, self.val_loader = self._initDataloader()  # A2D2
        self.prediction_path = os.path.join(self.setting.save_path, "preds")

        self.evaluator = pc_processor.metrics.IOUEval(
            n_classes=self.setting.n_classes, device=torch.device("cpu"), ignore=[0])
         

    def _initDataloader(self):
        if self.setting.dataset == "a2d2":
            camsLidars_path = "../../pc_processor/dataset/a2d2/cams_lidars.json"
            classIndex_path = "../../pc_processor/dataset/a2d2/class_index.json"
            dataset = pc_processor.dataset.a2d2.A2D2(
                root=self.setting.data_root,
                camsLidars_path=camsLidars_path,
                classIndex_path=classIndex_path,
                split='test'
            )
        else:
            raise ValueError(
                "invalid dataset: {}".format(self.setting.dataset))

        val_set = pc_processor.dataset.a2d2.A2D2EvalLoader(
            dataset=dataset,
            config=self.setting.config
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=1,
            num_workers=self.setting.n_threads,
            shuffle=False,
            drop_last=False
        )

        return val_set, val_loader


    def run(self):
        self.model.eval()
        self.evaluator.reset()

        with torch.no_grad():
            t_start = time.time()
            feature_mean = torch.Tensor(self.setting.config["PVconfig"]["pcd_mean"]).unsqueeze(
                0).unsqueeze(2).unsqueeze(2).cuda()
            feature_std = torch.Tensor(self.setting.config["PVconfig"]["pcd_stds"]).unsqueeze(
                0).unsqueeze(2).unsqueeze(2).cuda()
            data_len = len(self.val_set)

            for i, proj_data in enumerate(self.val_loader):
                t_process_start = time.time()

                input_feature = proj_data[:, 0:8].cuda()
                input_mask = proj_data[:, 8].cuda()
                input_label = proj_data[:, 9].long().cuda()

                input_feature[:, 0:5] = (input_feature[:, 0:5] - feature_mean) / feature_std * input_mask.unsqueeze(1).expand_as(input_feature[:, 0:5])
                pcd_feature = input_feature[:, 0:5]
                img_feature = input_feature[:, 5:8]

                if self.setting.net_type == "PMFNetV2":
                    pred_output, _ = self.model(pcd_feature, img_feature)
                else:
                    pred_output = self.model(pcd_feature)
                
                pred_argmax = pred_output[0].argmax(dim=0)
                argmax = pred_output.argmax(dim=1)

                if self.setting.has_label:
                    self.evaluator.addBatch(argmax, input_label)
                    iter_miou, _ = self.evaluator.getIoU()

                t_process_end = time.time()
                process_time = t_process_end - t_process_start
                data_time = t_process_end - t_start
                t_start = time.time()

                log_str = "Iter [{:04d}|{:04d}] Datatime: {:0.3f} ProcessTime: {:0.3f}".format(
                    i, data_len, data_time, process_time)
                if self.setting.has_label:
                    log_str += " meanIOU {:0.4f}".format(
                        iter_miou.cpu().item())
                print(log_str)

                if self.setting.is_debug:
                    break
        
        if not self.setting.has_label:
            return
        
        # after inference, show results(iou, acc, recall)
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
                cls_eval_table.add_row([i, self.val_set.dataset.mapped_class_name[i],
                    iou.item(), cls_acc[i].cpu().item(), cls_recall[i].cpu().item()])
                latext_str += " & {:0.1f}".format(iou * 100)
        latext_str += " & {:0.1f}".format(m_iou.cpu().item() * 100)

        self.recorder.logger.info(cls_eval_table)
        self.recorder.logger.info("---- Latext Format String -----")
        self.recorder.logger.info(latext_str)


        # conf_matrix
        conf_matrix = self.evaluator.conf_matrix.clone().cpu()
        conf_matrix[0] = 0
        conf_matrix[:, 0] = 0

        distribution_table = PrettyTable(["Class Name", "Number of points", "Percentage"])
        dist_data = conf_matrix.sum(0)
        for i in range(self.setting.n_classes):
            distribution_table.add_row([self.val_set.dataset.mapped_class_name[i], dist_data[i].item(), (dist_data[i]/dist_data.sum()).item()])
        self.recorder.logger.info("---- Data Distribution -----")
        self.recorder.logger.info(distribution_table)
        
        # fwIoU
        freqw = dist_data[1:] / dist_data[1:].sum()
        freq_iou = (cls_iou[1:] * freqw).sum()
        self.recorder.logger.info("fwIoU: {}".format(freq_iou.item()))
        
        self.recorder.logger.info("---- confusion matrix original data -----")
        self.recorder.logger.info(conf_matrix)

        # acc_matrix
        acc_data = conf_matrix.float() / (conf_matrix.sum(1, keepdim=True).float() + 1e-8)
        table_title = [" "]
        for i in range(1, self.setting.n_classes):
            table_title.append("{}".format(self.val_set.dataset.mapped_class_name[i]))
        acc_table = PrettyTable(table_title)
        for i in range(1, self.setting.n_classes):
            row_data = ["{}".format(self.val_set.dataset.mapped_class_name[i])]
            for j in range(1, self.setting.n_classes):
                row_data.append("{:0.1f}".format(acc_data[i, j]*100))
            acc_table.add_row(row_data)
        self.recorder.logger.info("---- ACC matrix ----------------")
        self.recorder.logger.info(acc_table)

        # recall_matrix
        recall_data = conf_matrix.float() / (conf_matrix.sum(0, keepdim=True).float()+1e-8)
        table_title = [" "]
        for i in range(1, self.setting.n_classes):
            table_title.append("{}".format(self.val_set.dataset.mapped_class_name[i]))
        recall_table = PrettyTable(table_title)
        for i in range(1, self.setting.n_classes):
            row_data = ["{}".format(self.val_set.dataset.mapped_class_name[i])]
            for j in range(1, self.setting.n_classes):
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
            raise NotImplementedError("invalid net_type: {}".format(self.settings.net_type))
        return model


    def _loadCheckpoint(self):
        if self.settings.pretrained_model is not None:
            if not os.path.isfile(self.settings.pretrained_model):
                raise FileNotFoundError("pretrained model not found: {}".format(self.settings.pretrained_model))
            state_dict = torch.load(self.settings.pretrained_model, map_location="cpu")
            # state_dict = {k: v for k, v in state_dict.items() if k in self.model.state_dict()}
            self.model.load_state_dict(state_dict)
            if self.recorder is not None:
                self.recorder.logger.info("loading pretrained weight from: {}".format(self.settings.pretrained_model))
            

    def run(self):
        t_start = time.time()
        self.inference.run()
        t_end = time.time()
        cost_time = t_end - t_start
        self.recorder.logger.info(
            "==== total cost time: {}".format(datetime.timedelta(seconds=cost_time))
        )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Options")
    parser.add_argument("--config_path", type=str, metavar="config_path",
                        help="path of config file, type: string")
    parser.add_argument("--id", type=int, metavar="experiment_id", required=False,
                        help="id of experiment", default=0)
    args = parser.parse_args()
    experiment = Experiment(Option(args.config_path))
    print("===init env success===")
    experiment.run()
