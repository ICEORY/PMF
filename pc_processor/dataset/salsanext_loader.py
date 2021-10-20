import numpy as np
import torch
from torch.utils.data import Dataset
from pc_processor.dataset.preprocess import augmentor, projection

class SalsaNextLoader(Dataset):
    def __init__(self, dataset, config, data_len=-1, is_train=True, return_uproj=False):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.data_len = data_len
        self.return_uproj = return_uproj

        augment_params = augmentor.AugmentParams()
        augment_config = self.config['augmentation']

        if self.is_train:
            augment_params.setFlipProb(
                p_flipx=augment_config['p_flipx'], p_flipy=augment_config['p_flipy'])
            augment_params.setTranslationParams(
                p_transx=augment_config['p_transx'], trans_xmin=augment_config[
                    'trans_xmin'], trans_xmax=augment_config['trans_xmax'],
                p_transy=augment_config['p_transy'], trans_ymin=augment_config[
                    'trans_ymin'], trans_ymax=augment_config['trans_ymax'],
                p_transz=augment_config['p_transz'], trans_zmin=augment_config[
                    'trans_zmin'], trans_zmax=augment_config['trans_zmax'])
            augment_params.setRotationParams(
                p_rot_roll=augment_config['p_rot_roll'], rot_rollmin=augment_config[
                    'rot_rollmin'], rot_rollmax=augment_config['rot_rollmax'],
                p_rot_pitch=augment_config['p_rot_pitch'], rot_pitchmin=augment_config[
                    'rot_pitchmin'], rot_pitchmax=augment_config['rot_pitchmax'],
                p_rot_yaw=augment_config['p_rot_yaw'], rot_yawmin=augment_config[
                    'rot_yawmin'], rot_yawmax=augment_config['rot_yawmax'])
            self.augmentor = augmentor.Augmentor(augment_params)

        else:
            self.augmentor = None

        projection_config = self.config['sensor']
        self.projection = projection.RangeProjection(
            fov_up=projection_config['fov_up'], fov_down=projection_config['fov_down'],
            fov_left=projection_config['fov_left'], fov_right=projection_config['fov_right'],
            proj_h=projection_config['proj_h'], proj_w=projection_config['proj_w'],
        )
        self.proj_img_mean = torch.tensor(self.config["sensor"]["img_mean"], dtype=torch.float)
        self.proj_img_stds = torch.tensor(self.config["sensor"]["img_stds"], dtype=torch.float)

    def __getitem__(self, index):
        """
        proj_feature_tensor: CxHxW
        proj_sem_label_tensor: HxW
        proj_mask_tensor: HxW
        """
        pointcloud, sem_label, inst_label = self.dataset.loadDataByIndex(index)
        if self.is_train:
            pointcloud = self.augmentor.doAugmentation(pointcloud)  # n, 4
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)

        proj_mask_tensor = torch.from_numpy(proj_mask)
        mask = proj_idx > 0
        proj_sem_label = np.zeros((proj_mask.shape[0], proj_mask.shape[1]), dtype=np.float32)
        proj_sem_label[mask] = self.dataset.labelMapping(sem_label[proj_idx[mask]])
        proj_sem_label_tensor = torch.from_numpy(proj_sem_label)
        proj_sem_label_tensor = proj_sem_label_tensor * proj_mask_tensor.float()

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0)], 0)

        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None,
                                                                                          None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        if self.return_uproj:
            uproj_x_tensor = torch.from_numpy(self.projection.cached_data["uproj_x_idx"]).long()
            uproj_y_tensor = torch.from_numpy(self.projection.cached_data["uproj_y_idx"]).long()
            uproj_depth_tensor = torch.from_numpy(self.projection.cached_data["uproj_depth"]).float()
            return proj_feature_tensor, proj_sem_label_tensor, proj_mask_tensor, torch.from_numpy(
                proj_range), uproj_x_tensor, uproj_y_tensor, uproj_depth_tensor
        else:
            return proj_feature_tensor, proj_sem_label_tensor, proj_mask_tensor

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)