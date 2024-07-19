import os
import yaml
import numpy as np
from PIL import Image


class SemanticKitti(object):
    def __init__(self, root,  # directory where data is
                 sequences,  # sequences for this data (e.g. [1,3,4,6])
                 config_path,  # directory of config file
                 has_image=True,
                 has_pcd=True,
                 has_label=True):
        self.root = root
        self.sequences = sequences
        self.sequences.sort()  # sort seq id
        self.has_label = has_label
        self.has_image = has_image
        self.has_pcd = has_pcd

        # check file exists
        if os.path.isfile(config_path):
            self.data_config = yaml.safe_load(open(config_path, "r"))
        else:
            raise ValueError("config file not found: {}".format(config_path))

        if os.path.isdir(self.root):
            print("Dataset found: {}".format(self.root))
        else:
            raise ValueError("dataset not found: {}".format(self.root))

        self.pointcloud_files = []
        self.label_files = []
        self.image_files = []
        self.proj_matrix = {}
        self.fov_left = -45 / 180.0 * np.pi
        self.fov_right = 45 / 180.0 * np.pi

        for seq in self.sequences:
            # format seq id
            seq = "{0:02d}".format(int(seq))
            print("parsing seq {}...".format(seq))

            # get file list from path
            pointcloud_path = os.path.join(self.root, seq, "velodyne")
            pointcloud_files = [os.path.join(pointcloud_path, f) for f in os.listdir(
                pointcloud_path) if ".bin" in f]

            if self.has_label:
                label_path = os.path.join(self.root, seq, "labels")
                label_files = [os.path.join(label_path, f)
                               for f in os.listdir(label_path) if ".label" in f]
            if self.has_image:
                image_path = os.path.join(self.root, seq, "image_2")
                image_files = [os.path.join(image_path, f) for f in os.listdir(
                    image_path) if ".png" in f]

            if self.has_pcd:
                if self.has_label:
                    assert (len(pointcloud_files) == len(label_files))
                if self.has_image:
                    assert (len(pointcloud_files) == len(image_files))

            self.pointcloud_files.extend(pointcloud_files)
            if self.has_label:
                self.label_files.extend(label_files)
            if self.has_image:
                self.image_files.extend(image_files)

            # load calibration file
            if self.has_image:
                calib_path = os.path.join(self.root, seq, "calib.txt")
                calib = self.read_calib(calib_path)
                proj_matrix = np.matmul(calib["P2"], calib["Tr"])
                self.proj_matrix[seq] = proj_matrix

        # sort for correspondance
        if self.has_pcd:
            self.pointcloud_files.sort()
        if self.has_label:
            self.label_files.sort()
        if self.has_image:
            self.image_files.sort()
        print("Using {} pointclouds from sequences {}".format(
            len(self.pointcloud_files), self.sequences))

        # load config -------------------------------------
        # get color map
        sem_color_map = self.data_config["color_map"]
        max_sem_key = 0
        for k, v in sem_color_map.items():
            if k + 1 > max_sem_key:
                max_sem_key = k + 1
        self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for k, v in sem_color_map.items():
            self.sem_color_lut[k] = np.array(v, np.float32) / 255.0

        sem_color_inv_map = self.data_config["color_map_inv"]
        max_sem_key = 0
        for k, v in sem_color_inv_map.items():
            if k + 1 > max_sem_key:
                max_sem_key = k + 1
        self.sem_color_lut_inv = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for k, v in sem_color_inv_map.items():
            self.sem_color_lut_inv[k] = np.array(v, np.float32) / 255.0

        self.inst_color_map = np.random.uniform(
            low=0.0, high=1.0, size=(10000, 3))

        # get learning class map
        # map unused classes to used classes
        learning_map = self.data_config["learning_map"]
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut[k] = v
        # learning map inv
        learning_map = self.data_config["learning_map_inv"]
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut_inv = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut_inv[k] = v

        # compute ignore class by content ratio
        cls_content = self.data_config["content"]
        content = np.zeros(len(self.data_config["learning_map_inv"]), dtype=np.float32)
        for cl, freq in cls_content.items():
            x_cl = self.class_map_lut[cl]
            content[x_cl] += freq
        self.cls_freq = content

        self.mapped_cls_name = self.data_config["mapped_class_name"]

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
        return calib_out

    @staticmethod
    def readPCD(path):
        pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return pcd

    @staticmethod
    def readLabel(path):
        label = np.fromfile(path, dtype=np.int32)
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        return sem_label, inst_label

    def parsePathInfoByIndex(self, index):
        path = self.pointcloud_files[index]
        # linux path
        if "\\" in path:
            # windows path
            path_split = path.split("\\")
        else:
            path_split = path.split("/")
        seq_id = path_split[-3]
        frame_id = path_split[-1].split(".")[0]
        return seq_id, frame_id

    def labelMapping(self, label):
        label = self.class_map_lut[label]
        return label

    def loadLabelByIndex(self, index):
        sem_label, inst_label = self.readLabel(self.label_files[index])
        return sem_label, inst_label

    def loadDataByIndex(self, index):
        pointcloud = self.readPCD(self.pointcloud_files[index])
        if self.has_label:
            sem_label, inst_label = self.readLabel(self.label_files[index])
        else:
            # return fake label
            sem_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
            inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label

    def loadImage(self, index):
        return Image.open(self.image_files[index])

    def mapLidar2Camera(self, seq, pointcloud, img_h, img_w):
        if not self.has_image:
            raise ValueError("cannot mappint pointcloud with has_image=False")

        proj_matrx = self.proj_matrix[seq]
        # only keep point in front of the vehicle
        keep_mask = pointcloud[:, 0] > 0.5
        pointcloud_hcoord = np.concatenate([pointcloud[keep_mask], np.ones(
            [keep_mask.sum(), 1], dtype=np.float32)], axis=1)
        mapped_points = (proj_matrx @ pointcloud_hcoord.T).T  # n, 3
        # scale 2D points
        mapped_points = mapped_points[:, :2] / \
                        np.expand_dims(mapped_points[:, 2], axis=1)  # n, 2
        keep_idx_pts = (mapped_points[:, 0] > 0) * (mapped_points[:, 0] < img_h) * (
                mapped_points[:, 1] > 0) * (mapped_points[:, 1] < img_w)
        keep_mask[keep_mask] = keep_idx_pts
        # fliplr so that indexing is row, col and not col, row
        mapped_points = np.fliplr(mapped_points)
        return mapped_points[keep_idx_pts], keep_mask
    
    def mapLidar2CameraCropYaw(self, seq, pointcloud):
        # map lidar to camera view and crop the image by yaw angle
        if not self.has_image:
            raise ValueError("cannot mappint pointcloud with has_image=False")
        
        # only keep point in front of the vehicle
        # only keep the points outside 0.5 meters
        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        depth_keep_mask = depth > 0.5

        # only keep the points within [fov_left, fov_right] in yaw
        crop_pointcloud = pointcloud.copy()
        yaw = -np.arctan2(crop_pointcloud[:, 1], crop_pointcloud[:, 0])
        fov_keep_mask = (yaw >= self.fov_left) * (yaw <= self.fov_right)
        keep_mask = np.logical_and(depth_keep_mask, fov_keep_mask)
        crop_pointcloud = crop_pointcloud[keep_mask]

        proj_matrx = self.proj_matrix[seq]
        
        pointcloud_hcoord = np.concatenate([crop_pointcloud[:, :3], np.ones(
            [keep_mask.sum(), 1], dtype=np.float32)], axis=1)
        mapped_points = (proj_matrx @ pointcloud_hcoord.T).T  # n, 3
        # scale 2D points
        mapped_points = mapped_points[:, :2] / \
                        np.expand_dims(mapped_points[:, 2], axis=1)  # n, 2
        # fliplr so that indexing is row, col and not col, row
        mapped_points = np.fliplr(mapped_points)

        return crop_pointcloud, mapped_points, keep_mask
        

    def __len__(self):
        return len(self.pointcloud_files)