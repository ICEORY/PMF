import math
from cv2 import rotate
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

class PerspectiveViewLoaderV2(Dataset):
    def __init__(self, dataset, config, data_len=-1, is_train=True, img_aug=False,
                 return_uproj=False):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.img_aug = img_aug
        self.data_len = data_len
        self.pv_config = self.config["PVconfig"]

        if self.img_aug:
            self.img_jitter = transforms.ColorJitter(
                *self.pv_config["img_jitter"])
        else:
            self.img_jitter = None

        if self.is_train:
            # if training, random scale
            # h_flip, random rotation, random crop
            self.aug_ops = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.RandomCrop(
                    size=(self.pv_config["proj_ht"],
                          self.pv_config["proj_wt"])),
            ])
        else:
            self.aug_ops = transforms.Compose([
                transforms.CenterCrop((self.pv_config["proj_h"],
                                       self.pv_config["proj_w"]))
            ])
        self.return_uproj = return_uproj

    def __getitem__(self, index):
        # feature: range, x, y, z, i, rgb
        # get image feature
        image = self.dataset.loadImage(index)
        if self.img_aug:
            image = self.img_jitter(image)
        
        # random scale
        img_w, img_h = image.size
        if self.is_train:
            img_scale = np.random.uniform(low=1.0, high=1.2)
            scale_op = transforms.Resize(
                size=(int(img_h*img_scale), int(img_w*img_scale)))
            image = scale_op(image)
        else:
            img_scale = 1

        image = np.array(image)
        image = image.astype(np.float32) / 255.0

        # get point cloud and label
        pointcloud, sem_label, _ = self.dataset.loadDataByIndex(index)
        seq_id, _ = self.dataset.parsePathInfoByIndex(index)
        
        if self.is_train:
            # random drop
            max_h, max_w = self.pv_config["proj_ht"], self.pv_config["proj_wt"] # 0, 0
        else:
            max_h, max_w = self.pv_config["proj_h"], self.pv_config["proj_w"]

        crop_pointcloud, xy_index, keep_mask = self.dataset.mapLidar2CameraCropYaw(
            seq_id, pointcloud
        )
        xy_index = xy_index*img_scale

        sem_label = sem_label[keep_mask]
            
        x_data = xy_index[:, 0].astype(np.int32)
        y_data = xy_index[:, 1].astype(np.int32)
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()

        h, w = x_max-x_min+1, y_max-y_min+1
        # print(h, w)
        if h > max_h:
            max_h = h 
        if w > max_w:
            max_w = w

        proj_xyzi = np.zeros(
            (h, w, crop_pointcloud.shape[1]), dtype=np.float32)
        proj_xyzi[x_data-x_min, y_data-y_min] = crop_pointcloud
        
        proj_depth = np.zeros((h, w), dtype=np.float32)
        # compute image view pointcloud feature
        depth = np.linalg.norm(crop_pointcloud[:, :3], 2, axis=1)
        proj_depth[x_data-x_min, y_data-y_min] = depth
        
        proj_label = np.zeros((h, w), dtype=np.int32)
        proj_label[x_data-x_min, y_data-y_min] = self.dataset.labelMapping(sem_label)
        
        proj_mask = np.zeros((h, w), dtype=np.int32)
        proj_mask[x_data-x_min, y_data-y_min] = 1


        proj_rgb = np.zeros((h, w, 3), dtype=np.float32)
        if x_min >= 0:
            px_start = 0
            px_end = image.shape[0] - x_min
            ix_start = x_min
        else:
            px_start = -x_min
            px_end = image.shape[0] - x_min
            ix_start = 0

        if y_min >= 0:
            py_start = 0
            py_end = image.shape[1] - y_min
            iy_start = y_min
        else:
            py_start = -y_min
            py_end = image.shape[1] - y_min
            iy_start = 0
        px_end = min(px_end, proj_rgb.shape[0])
        py_end = min(py_end, proj_rgb.shape[1])
        if py_end > 0 and px_end > 0:
            proj_rgb[px_start:px_end, py_start:py_end] = image[ix_start:ix_start+px_end-px_start, iy_start:iy_start+py_end-py_start]

        # convert data to tensor 
        proj_tensor = torch.cat(
            (
                torch.from_numpy(proj_depth).unsqueeze(0),
                torch.from_numpy(proj_xyzi).permute(2, 0, 1),
                torch.from_numpy(proj_rgb).permute(2, 0, 1),
                torch.from_numpy(proj_mask).float().unsqueeze(0),
                torch.from_numpy(proj_label).float().unsqueeze(0)
            ), dim=0
        )
        xy_index_tensor = torch.from_numpy(xy_index.copy())
        depth_tensor = torch.from_numpy(depth)
        keep_mask_tensor = torch.from_numpy(keep_mask)
            
        if self.return_uproj:
            return proj_tensor, xy_index_tensor, depth_tensor, keep_mask_tensor, torch.from_numpy(pointcloud)
        else:
            h_pad = max_h - proj_tensor.size(1)
            w_pad_left = (max_w - proj_tensor.size(2)) // 2
            w_pad_right = max_w - proj_tensor.size(2) - w_pad_left

            pad_op = transforms.Pad((w_pad_left, 0, w_pad_right, h_pad))
            pad_proj_tensor = pad_op(proj_tensor)
            # tensor augmentation
            if self.aug_ops is not None:
                pad_proj_tensor = self.aug_ops(pad_proj_tensor)

            return pad_proj_tensor

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)
