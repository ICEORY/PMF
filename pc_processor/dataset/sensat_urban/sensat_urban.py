import os
import torch 
import math
import numpy as np
from tqdm import trange

class SensatUrban(object):
    def __init__(self, root_path, split="train", keep_idx=False, img_h=320, img_w=320, use_crop=False):
        self.root_path = root_path
        self.split = split
        self.keep_idx = keep_idx
        self.img_h = img_h
        self.img_w = img_w 
        self.use_crop = use_crop

        if self.split not in ["train", "test", "val"]:
            raise ValueError("invalid split: {}".format(self.split))
        self.split_folder = os.path.join(self.root_path, self.split) 
        self.data_split = []
        for file_name in os.listdir(self.split_folder):
            if ".pth" in file_name and "cambridge_block_1" not in file_name:
                # skip cambridge_block_1 (tiny block) to avoid error
                self.data_split.append(file_name)
                # if split == "train":
                # break

        self.all_data_frame = self.loadDataCache()
        print("Using {} data frame from {} split".format(
            len(self.all_data_frame), self.split
        ))

        self.mapped_cls_name = {
            -1: "ignore", 
            0: 'Ground', 1: 'High Vegetation', 2: 'Buildings', 3: 'Walls',
            4: 'Bridge', 5: 'Parking', 6: 'Rail', 7: 'traffic Roads', 8: 'Street Furniture',
            9: 'Cars', 10: 'Footpath', 11: 'Bikes', 12: 'Water'
        }

    def loadDataCache(self):
        all_data_frame = []
        print("loading data frame...")
        n_frame = len(self.data_split)
        for i in trange(n_frame):
            file_name = self.data_split[i]
            data_frame = torch.load(os.path.join(self.split_folder, file_name))
            if not self.keep_idx:
                data_frame["h_idx"] = None
                data_frame["w_idx"] = None
            
            if self.use_crop:
                h = data_frame["feature_map"].shape[1]
                w = data_frame["feature_map"].shape[2]
                rows = math.ceil(h/ self.img_h)
                cols = math.ceil(w / self.img_w)
                for r in range(rows):
                    h_start = r * self.img_h 
                    h_end = (r + 1) * self.img_h
                    if h_end > h:
                        h_end = h
                        h_start = h - self.img_h 
                        if h_start < 0:
                            h_start = 0
                    
                    for c in range(cols):
                        w_start = c * self.img_w
                        w_end = (c + 1) * self.img_w
                        if w_end > w:
                            w_end = w
                            w_start = w - self.img_w 
                            if w_start < 0:
                                w_start = 0
                        new_feature_map = np.zeros((8, self.img_h, self.img_w))
                        new_feature_map[:, :h_end-h_start, :w_end-w_start] = data_frame["feature_map"][:, h_start:h_end, w_start:w_end]
                        new_label_map = np.zeros((self.img_h, self.img_w))
                        new_label_map[:h_end-h_start, :w_end-w_start] = data_frame["label_map"][h_start:h_end, w_start:w_end]

                        tmp_data_frame = {
                            "feature_map": new_feature_map,
                            "label_map": new_label_map,
                        }
                        all_data_frame.append(tmp_data_frame)

            else:
                all_data_frame.append(data_frame)
        return all_data_frame

    def readLabelByIndex(self, index):
        label_path = os.path.join(self.split_folder, self.data_split[index].replace(".pth", ".bin"))
        label = np.fromfile(label_path, dtype=np.uint8)
        return label 
    
    def readFileNameByIndex(self, index):
        label_path = self.data_split[index].replace(".pth", ".bin")
        return label_path

    def readDataByIndex(self, index):
        return self.all_data_frame[index]

    def __len__(self):
        return len(self.all_data_frame)