# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_lidarseg_tutorial.ipynb

import os
import numpy as np
import yaml
# import cv2
from pathlib import Path
from torch.utils import data
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud  # , Box
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, plt_to_cv2, get_stats, \
    get_labels_in_coloring, create_lidarseg_legend, paint_points_label

map_name_from_general_to_segmentation_class = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
    'flat.driveable_surface': 'driveable_surface',
    'flat.other': 'other_flat',
    'flat.sidewalk': 'sidewalk',
    'flat.terrain': 'terrain',
    'static.manmade': 'manmade',
    'static.vegetation': 'vegetation',
    'noise': 'ignore',
    'static.other': 'ignore',
    'vehicle.ego': 'ignore'
}

map_name_from_segmentation_class_to_segmentation_index = {
    'ignore': 0,
    'barrier': 1,
    'bicycle': 2,
    'bus': 3,
    'car': 4,
    'construction_vehicle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'trailer': 9,
    'truck': 10,
    'driveable_surface': 11,
    'other_flat': 12,
    'sidewalk': 13,
    'terrain': 14,
    'manmade': 15,
    'vegetation': 16
}


class Nuscenes(data.Dataset):
    def __init__(self, root,
                 version='v1.0-trainval',
                 split='train',
                 return_ref=False,
                 has_image=True,
                 has_pcd=True,
                 has_label=True):

        assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
        if version == 'v1.0-trainval':
            train_scenes = splits.train
        elif version == 'v1.0-test':
            train_scenes = splits.test
        elif version == 'v1.0-mini':
            train_scenes = splits.mini_train
        else:
            raise NotImplementedError
        self.split = split
        self.data_path = root
        self.return_ref = return_ref

        self.nusc = NuScenes(
            version=version, dataroot=self.data_path, verbose=False)
        self.has_image = has_image

        self.map_name_from_general_index_to_segmentation_index = {}
        for index in self.nusc.lidarseg_idx2name_mapping:
            self.map_name_from_general_index_to_segmentation_index[index] = \
                map_name_from_segmentation_class_to_segmentation_index[
                    map_name_from_general_to_segmentation_class[self.nusc.lidarseg_idx2name_mapping[index]]]

        self.mapped_cls_name = {}
        for v, k in map_name_from_segmentation_class_to_segmentation_index.items():
            self.mapped_cls_name[k] = v

        available_scenes = get_available_scenes(self.nusc)
        available_scene_names = [s['name'] for s in available_scenes]
        train_scenes = list(
            filter(lambda x: x in available_scene_names, train_scenes))
        train_scenes = set(
            [available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])

        if self.has_image:
            train_token_list, val_token_list = get_path_infos_cam_lidar(
                self.nusc, train_scenes)
        else:
            train_token_list, val_token_list = get_path_infos_only_lidar(
                self.nusc, train_scenes)

        if self.split == "train" or self.split == "test":
            self.token_list = train_token_list
        elif self.split == "val":
            self.token_list = val_token_list
        else:
            raise ValueError("invalid split mode: {}".format(self.split))
        print("{}: {} sample: {}".format(
            version, self.split, len(self.token_list)))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.token_list)

    def parsePathInfoByIndex(self, index):
        return index, ''

    def loadLabelByIndex(self, index):
        if self.has_image:
            lidar_sample_token = self.token_list[index]['lidar_token']
        else:
            lidar_sample_token = self.token_list[index]

        if self.split == 'test':
            self.lidarseg_path = None
            annotated_data = None
        else:
            lidarseg_path = os.path.join(self.data_path,
                                         self.nusc.get('lidarseg', lidar_sample_token)['filename'])
            annotated_data = np.fromfile(
                lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label
        return annotated_data

    def loadDataByIndex(self, index):
        if self.has_image:
            lidar_sample_token = self.token_list[index]['lidar_token']
        else:
            lidar_sample_token = self.token_list[index]

        lidar_path = os.path.join(self.data_path,
                                  self.nusc.get('sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.split == 'test':
            self.lidarseg_path = None
            annotated_data = np.expand_dims(
                np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            lidarseg_path = os.path.join(self.data_path,
                                         self.nusc.get('lidarseg', lidar_sample_token)['filename'])
            annotated_data = np.fromfile(
                lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label

        pointcloud = raw_data[:, :4]
        sem_label = annotated_data
        inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label

    def labelMapping(self, sem_label):
        sem_label = np.vectorize(self.map_name_from_general_index_to_segmentation_index.__getitem__)(
            sem_label)  # n, 1
        assert sem_label.shape[-1] == 1
        sem_label = sem_label[:, 0]
        return sem_label

    def loadImage(self, index):
        cam_sample_token = self.token_list[index]['cam_token']
        cam = self.nusc.get('sample_data', cam_sample_token)
        image = Image.open(os.path.join(self.nusc.dataroot, cam['filename']))
        return image

    def getColorMap(self):
        '''
        useage: coloring = colors[points_label]
        :return: A numpy array which has length equal to the number of points in the pointcloud, and each value is
             a RGBA array.
        '''
        colors = colormap_to_colors(
            self.nusc.colormap, self.nusc.lidarseg_name2idx_mapping)  # Shape: [num_class, 3]
        return colors

    def mapLidar2Camera(self,
                        index,
                        pointcloud,
                        img_h,
                        img_w,
                        min_dist: float = 1.0,
                        show_lidarseg: bool = True,  # False
                        render_intensity: bool = True,  # False
                        filter_lidarseg_labels=None,
                        vis_render_img=False
                        ):
        lidar_sample_token = self.token_list[index]['lidar_token']
        pointsensor = self.nusc.get('sample_data', lidar_sample_token)

        assert pointsensor['is_key_frame'], \
            'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, ' \
            'not %s!' % pointsensor['sensor_modality']

        # Projects a pointcloud into a camera image along with the lidarseg labels
        cam_sample_token = self.token_list[index]['cam_token']
        cam = self.nusc.get('sample_data', cam_sample_token)
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor['filename'])

        pc = LidarPointCloud.from_file(pcl_path)

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.nusc.get(
            'calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.nusc.get(
            'calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]
        if depths.shape[0] < 10000:
            print(depths.shape)
            print(pc.points.shape)
            print(pcl_path)

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(
            cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        # tmp_mask = depths < min_dist
        # print("tmp_mask sum: ", tmp_mask.sum(), depths.shape)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < img_h - 1)
        # mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < img_w - 1)

        mapped_points = points.transpose(1, 0)  # n, 3
        mapped_points = np.fliplr(mapped_points[:, :2])

        # fliplr so that indexing is row, col and not col, row
        return mapped_points[mask, :], mask  # (3, n) (n, )


def get_available_scenes(nusc):
    # only for check if all the files are available
    available_scenes = []
    # print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, _, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break

        if scene_not_exist:
            continue
        available_scenes.append(scene)
    # print('exist scene num:', len(available_scenes))
    return available_scenes


def get_path_infos_only_lidar(nusc, train_scenes):
    train_lidar_token_list = []
    val_lidar_token_list = []
    for sample in nusc.sample:
        scene_token = sample['scene_token']
        lidar_token = sample['data']['LIDAR_TOP']

        if scene_token in train_scenes:
            train_lidar_token_list.append(lidar_token)
        else:
            val_lidar_token_list.append(lidar_token)
    print(len(train_lidar_token_list), len(val_lidar_token_list))
    return train_lidar_token_list, val_lidar_token_list


def get_path_infos_cam_lidar(nusc, train_scenes):
    train_lidar_token_list = []
    val_lidar_token_list = []

    for sample in nusc.sample:
        scene_token = sample['scene_token']
        lidar_token = sample['data']['LIDAR_TOP']  # 360°的lidar

        if scene_token in train_scenes:
            for i in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']:
                a = {'lidar_token': lidar_token,
                     'cam_token': sample['data'][i]}
                train_lidar_token_list.append(a)
        else:
            for i in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']:
                a = {'lidar_token': lidar_token,
                     'cam_token': sample['data'][i]}
                val_lidar_token_list.append(a)
    # print(len(train_lidar_token_list), len(val_lidar_token_list))
    return train_lidar_token_list, val_lidar_token_list


if __name__ == '__main__':
    # data_path = '/mnt/cephfs/dataset/pointclouds/nuscenes/lidarseg'
    data_path = '/mnt/cephfs/dataset/pointclouds/nuscenes'
    # dataset = Nuscenes(root=data_path, version='v1.0-trainval', split='train', return_ref=False)
    dataset = Nuscenes(root='/mnt/cephfs/home/lirong/data/nuscenes', version='v1.0-mini', split='train',
                       return_ref=False)
    data = dataset.loadDataByIndex(index=10)
