import numpy as np
import torch
from torch.utils.data import Dataset

class NusPerspectiveViewLoader(Dataset):
    def __init__(self, dataset, config, data_len=-1):
        self.dataset = dataset
        self.config = config
        self.data_len = data_len

    def __getitem__(self, index):
        # feature: range, x, y, z, i, rgb
        pointcloud, sem_label, _ = self.dataset.loadDataByIndex(index)
        
        # get image feature
        image = self.dataset.loadImage(index)

        image = np.array(image)
        seq_id, _ = self.dataset.parsePathInfoByIndex(index)
        try:
            mapped_pointcloud, keep_mask = self.dataset.mapLidar2Camera(
                seq_id, pointcloud[:, :3], image.shape[1], image.shape[0])
        except Exception as msg:
            print(msg)
            cam_sample_token = self.dataset.token_list[index]['cam_token']
            cam = self.dataset.nusc.get('sample_data', cam_sample_token)
            print(cam['filename'])
            print(image.shape)
            print(self.dataset.token_list[index]["lidar_token"])
            print(pointcloud.shape)

        y_data = mapped_pointcloud[:, 1].astype(np.int32)
        x_data = mapped_pointcloud[:, 0].astype(np.int32)

        image = image.astype(np.float32) / 255.0
        # compute image view pointcloud feature
        pointcloud_idx = np.arange(pointcloud.shape[0])
        pointcloud_idx_keep = pointcloud_idx[keep_mask]

        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        keep_poincloud = pointcloud[keep_mask]
        proj_xyzi = np.zeros(
            (image.shape[0], image.shape[1], keep_poincloud.shape[1]), dtype=np.float32)
        proj_xyzi[x_data, y_data] = keep_poincloud
        proj_depth = np.zeros(
            (image.shape[0], image.shape[1]), dtype=np.float32)
        proj_depth[x_data, y_data] = depth[keep_mask]

        proj_label = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)

        proj_label[x_data,  y_data] = self.dataset.labelMapping(sem_label[keep_mask])

        proj_mask = np.zeros(
            (image.shape[0], image.shape[1]), dtype=np.int32)
        proj_mask[x_data, y_data] = 1

        # convert data to tensor
        image_tensor = torch.from_numpy(image)
        proj_depth_tensor = torch.from_numpy(proj_depth)
        proj_xyzi_tensor = torch.from_numpy(proj_xyzi)
        proj_label_tensor = torch.from_numpy(proj_label)
        proj_mask_tensor = torch.from_numpy(proj_mask)

        proj_tensor = torch.cat(
            (proj_depth_tensor.unsqueeze(0),
             proj_xyzi_tensor.permute(2, 0, 1),
             image_tensor.permute(2, 0, 1),
             proj_mask_tensor.float().unsqueeze(0),
             proj_label_tensor.float().unsqueeze(0)), dim=0)

        return proj_tensor[:8], proj_tensor[8], proj_tensor[9], torch.from_numpy(x_data), torch.from_numpy(
            y_data), torch.from_numpy(depth[keep_mask]), torch.from_numpy(pointcloud_idx_keep), torch.Tensor([pointcloud.shape[0]])
        
    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)