import os 
import numpy as np
import math
import sensat_tools
import torch
import time
from tqdm import trange

class SenSatPreProcess(object):
    def __init__(self, root_path, split= "train", save_path="./"):
        self.root_path = root_path
        self.split = split
        if self.split not in ["train", "val"]:
            raise ValueError("invalid split: {}".format(self.split))

        self.data_split_folder = os.path.join(self.root_path, self.split)
        self.data_split = []
        for file_name in os.listdir(self.data_split_folder):
            if ".ply" in file_name:
                self.data_split.append(file_name)
        self.save_path = save_path

    def run(self):
        for i, file_name in enumerate(self.data_split):
            t_start = time.time()
            origin_ply_path = os.path.join(self.data_split_folder, file_name)
            data = sensat_tools.read_ply(filename=origin_ply_path)
            pointcloud_label = data["class"].astype(np.uint8) # shape = (n, )
            save_path = os.path.join(self.data_split_folder, file_name.replace(".ply", ".bin"))
            pointcloud_label.tofile(save_path)
            print("[{}] cost time: {}".format(i, time.time() - t_start))

if __name__ == "__main__":
    data_root = "/path/to/your/sensat-urban/"
    data_preprocess = SenSatPreProcess(data_root, split="val")
    data_preprocess.run()