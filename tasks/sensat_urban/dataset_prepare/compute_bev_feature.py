import os 
import numpy as np
import math
import sensat_tools
import torch
import time
from tqdm import trange

class SenSatPreProcess(object):
    def __init__(self, root_path, grid_size=0.1, split= "train", save_path="./"):
        self.root_path = root_path
        self.grid_size = grid_size
        self.split = split
        if self.split not in ["train", "val", "test"]:
            raise ValueError("invalid split: {}".format(self.split))
        if self.split == "test":
            self.has_label = False
        else:
            self.has_label = True

        self.data_split_folder = os.path.join(self.root_path, self.split)
        self.data_split = []
        for file_name in os.listdir(self.data_split_folder):
            if ".ply" in file_name:
                self.data_split.append(file_name)
        self.save_path = save_path


    def _compute_feature(self, pointcloud):
        # y:h, x: w
        # feature: max_h, min_h, mean_h, log(density), nonzero_mask, mean_r, mean_g, mean_b
        min_x = pointcloud[:, 0].min()
        min_y = pointcloud[:, 1].min()

        h_idx = ((pointcloud[:, 1] - min_y) / self.grid_size).astype(np.int32)
        w_idx = ((pointcloud[:, 0] - min_x) / self.grid_size).astype(np.int32)
        height = h_idx.max() + 1
        width = w_idx.max() + 1
        print(height, width)

        feature_map = np.zeros((8, height, width))
        label_map = np.zeros((height, width)) - 1
        
        n_points = pointcloud.shape[0]
        for i in trange(n_points):
            pc = pointcloud[i]
            if feature_map[4, h_idx[i], w_idx[i]]:
                if feature_map[0, h_idx[i], w_idx[i]] < pc[2]:
                    feature_map[0, h_idx[i], w_idx[i]] = pc[2]
                    label_map[h_idx[i], w_idx[i]] = pc[6]
                    # RGB
                    feature_map[5, h_idx[i], w_idx[i]] = pc[3]
                    feature_map[6, h_idx[i], w_idx[i]] = pc[4]
                    feature_map[7, h_idx[i], w_idx[i]] = pc[5]

                if feature_map[1, h_idx[i], w_idx[i]] > pc[2]:
                    feature_map[1, h_idx[i], w_idx[i]] = pc[2]   
            else:
                feature_map[0, h_idx[i], w_idx[i]] = pc[2]
                feature_map[1, h_idx[i], w_idx[i]] = pc[2]
                label_map[h_idx[i], w_idx[i]] = pc[6]
                # RGB
                feature_map[5, h_idx[i], w_idx[i]] = pc[3]
                feature_map[6, h_idx[i], w_idx[i]] = pc[4]
                feature_map[7, h_idx[i], w_idx[i]] = pc[5]

            feature_map[2, h_idx[i], w_idx[i]] += pc[2]
            feature_map[3, h_idx[i], w_idx[i]] += 1
            feature_map[4, h_idx[i], w_idx[i]] = 1



        feature_map[2] = feature_map[2] / (feature_map[3] + 1e-6)
        feature_map[3] = np.log10(feature_map[3] + 1)

        return {
            "feature_map": feature_map,
            "label_map": label_map,
            "h_idx": h_idx,
            "w_idx": w_idx,
        }

    def run(self):
        for i, file_name in enumerate(self.data_split):
            t_start = time.time()
            origin_ply_path = os.path.join(self.data_split_folder, file_name)
            data = sensat_tools.read_ply(filename=origin_ply_path)
            if self.has_label:
                pointcloud_label = data["class"] # shape = (n, )
            else:
                pointcloud_label = np.zeros(data["x"].shape)
            pointcloud = np.vstack((
                data["x"], data["y"], data["z"], 
                data["red"], data["green"], data["blue"], pointcloud_label)).T # shape = (n, 7)
            result_dict = self._compute_feature(pointcloud)
            torch.save(result_dict, os.path.join(self.root_path, self.split, file_name.replace(".ply", ".pth")))
            print("[{}] cost time: {}".format(i, time.time() - t_start))

if __name__ == "__main__":
    data_root = "/path/to/your/sensat-urban/"
    data_preprocess = SenSatPreProcess(data_root, split="train")
    data_preprocess.run()
    # data_preprocess = SenSatPreProcess(data_root, split="val")
    # data_preprocess.run()
    # data_preprocess = SenSatPreProcess(data_root, split="test")
    # data_preprocess.run()