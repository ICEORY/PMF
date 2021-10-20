import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np 
import math

class SensatLoader(Dataset):
    def __init__(self, dataset, img_h=800, img_w=800, n_samples_split=200):
        self.dataset = dataset
        self.img_h = img_h
        self.img_w = img_w 
        # self.downscale = downscale

        self.split = dataset.split
        if self.split == "train":
            self.aug_pos = transforms.Compose([
                transforms.RandomCrop(
                    size=(self.img_h*2, self.img_w*2)
                ),
                transforms.RandomRotation(360),
                transforms.RandomCrop(
                    size=(self.img_h, self.img_w)
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5)
            ])

            self.frame_idx_list = []
            for i in range(len(self.dataset)):
                data_frame = self.dataset.readDataByIndex(i)
                h = data_frame["feature_map"].shape[1]
                w = data_frame["feature_map"].shape[2]
                weighted_sample_num = int(n_samples_split * h/4000 * w/4000)
                self.frame_idx_list += [i] * weighted_sample_num

            self.total_samples = len(self.frame_idx_list)
            self.is_train = True
        else:
            self.aug_pos = None
            self.total_samples = len(self.dataset)
            self.is_train = False
        print("Generate {} samples from {} split".format(self.total_samples, self.split))

    def __getitem__(self, index):
        # FEATURE: max_h, min_h, mean_h, density, nonzero_mask, mean_r, mean_g, mean_b
        # Label: label

        # get frame idx
        if self.split == "train":
            # frame_idx = index % len(self.dataset)
            frame_idx = self.frame_idx_list[index]
            data_frame = self.dataset.readDataByIndex(frame_idx)
        else:
            data_frame = self.dataset.readDataByIndex(index)

        feature_map = torch.from_numpy(data_frame["feature_map"]).float()
        

        label_map = torch.from_numpy(data_frame["label_map"]).float()

        all_map = torch.cat((feature_map, label_map.unsqueeze(0)), dim=0)
        
        if self.is_train:
            valid_percent = 0
            while valid_percent < 0.1:
                tmp_map = self.aug_pos(all_map)
                valid_percent = tmp_map[8].ge(0).sum().float() / tmp_map[8].numel()
            all_map = tmp_map
            all_map[5:8] = (all_map[5:8]+random.uniform(-0.2, 0.2)) * all_map[4:5]
            all_map[0:3] = (all_map[0:3]+random.uniform(-2, 2)) * all_map[4:5]
        
        return all_map[:8], all_map[8]
        

    def __len__(self):
        return self.total_samples
