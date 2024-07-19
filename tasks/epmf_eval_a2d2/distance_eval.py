import numpy as np
import os
import glob
import torch
import sys
from tqdm import tqdm
sys.path.insert(0, "../../")
import pc_processor


class Options(object):
    def __init__(self):
        # self.debug = False
        self.preds_path = "/mnt/cephfs/home/chensitao/code/PMFv2/experiments/PMF-a2d2/log_a2d2_PMFNetV2-resnet34_bs12-lr0.00028_E150-MTL-ASPP-20230121-Epoch114_150/Eval-a2d2-PMFNet-best_IOU_model--test_20230129/"
        # self.seqs = [8]
        self.data_root = "/mnt/cephfs/dataset/pointclouds/a2d2/camera_lidar_semantic/"
        # self.data_config_path = "../../../pc_processor/dataset/semantic_kitti/semantic-kitti.yaml"
        self.camsLidars_path = "../../pc_processor/dataset/a2d2/cams_lidars.json"
        self.classIndex_path = "../../pc_processor/dataset/a2d2/class_index.json"

        self.distance_step = [10, 20, 30, 40, 50, 60, 125]
        self.n_classes = 39
        self.ignore_class = [0]
        self.save_path = os.path.join(self.preds_path, "log/distance_eval.log")


class DistanceEval(object):
    def __init__(self, settings: Options):
        self.settings = settings

        # init dataset
        self.dataset = pc_processor.dataset.a2d2.A2D2_PV(
            root=self.settings.data_root,
            camsLidars_path=self.settings.camsLidars_path,
            classIndex_path=self.settings.classIndex_path,
            split='test'
        )

        # init pred_files
        self.pred_files = sorted(glob.glob(os.path.join(self.settings.preds_path, 'preds/camera_lidar_semantic/*.label')))
        print(len(self.dataset), len(self.pred_files))
        assert len(self.pred_files) == len(self.dataset), "Error: number of files must be the same"

        # init dataset
        # self.dataset = pc_processor.dataset.semantic_kitti.SemanticKitti(
        #     root=self.settings.data_root,
        #     sequences=self.settings.seqs,
        #     config_path=self.settings.data_config_path,
        #     has_image=False, has_pcd=True, has_label=True
        # )

        # init metrics
        self.metrics_list = []
        for _ in range(len(self.settings.distance_step)):
            metrics = pc_processor.metrics.IOUEval(
                n_classes=self.settings.n_classes, device=torch.device("cpu"),
                ignore=self.settings.ignore_class, is_distributed=False)
            metrics.reset()
            self.metrics_list.append(metrics)
    
    def readLabel(self, path):   
        label = np.fromfile(path, dtype=np.int32)
        sem_label = label & 0xFFFF                   # semantic label in lower half
        inst_label = label >> 16                     # instance id in upper half
        return sem_label, inst_label

    def run(self):
        num_valid_labels = np.zeros(len(self.settings.distance_step))
        for i in tqdm(range(len(self.dataset)), "Processing"):
            # index, _ = self.dataset.parsePathInfoByIndex(i)
            # print(seq, frame_id) 
            # pred_file = os.path.join(self.settings.preds_path,
            #                          "preds/camera_lidar_semantic/{}.label".format(seq, frame_id))
            pred_file = self.pred_files[i]
            if not os.path.isfile(pred_file):
                raise FileNotFoundError(pred_file)

            pcd, sem_label, _ = self.dataset.loadDataByIndex(i)
            # sem_label = self.dataset.labelMapping(sem_label)

            sem_pred, _ = self.readLabel(pred_file)
            # sem_pred = self.dataset.labelMapping(sem_pred)

            dist = np.linalg.norm(pcd[:, :2], 2, axis=1)
            dist_min = 0
            for j in range(len(self.settings.distance_step)):
                dist_max = self.settings.distance_step[j]
                valid_mask = np.logical_and(dist < dist_max, dist >= dist_min)
                # mask label
                masked_sem_label = sem_label * valid_mask
                masked_sem_pred = sem_pred * valid_mask
                num_valid_labels[j] += valid_mask.sum()
                self.metrics_list[j].addBatch(masked_sem_pred, masked_sem_label)
                dist_min = dist_max
                # break

        # record result
        fig_log_str = "result for paper fig:  "
        with open(self.settings.save_path, "w") as f:
            f.write("distance mIOU mAcc mRecall\n")
            dist_min = 0
            total_valid_labels = num_valid_labels.sum()
            for j in range(len(self.settings.distance_step)):
                mean_iou, _ = self.metrics_list[j].getIoU()
                mean_acc, _ = self.metrics_list[j].getAcc()
                mean_recall, _ = self.metrics_list[j].getRecall()
                log_str = "{}-{} {:0.6f} {:0.6f} {:0.6f} {} {:0.4f}%\n".format(
                    dist_min, self.settings.distance_step[j], mean_iou.item(),
                    mean_acc.item(), mean_recall.item(), num_valid_labels[j], num_valid_labels[j]/total_valid_labels)
                print(log_str)
                f.write(log_str)
                dist_min = self.settings.distance_step[j]
                fig_log_str += ", {:0.6f}".format(mean_iou.item())
            f.write(fig_log_str)


if __name__ == "__main__":
    settings = Options()
    eval = DistanceEval(settings)
    eval.run()