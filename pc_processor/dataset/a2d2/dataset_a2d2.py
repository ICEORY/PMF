import json
import os
import glob
import cv2
from torch.utils import data
import numpy as np
from PIL import Image


mapped_class_name = {
    0: 'ignore',
    1: 'car',
    2: 'bicycle',
    3: 'pedestrian',
    4: 'truck',
    5: 'small_vehicles',
    6: 'traffic_signal',
    7: 'traffic_sign',
    8: 'utility_vehicle',
    9: 'sidebars',
    10: 'speed_bumper',
    11: 'curbstone',
    12: 'solid_line',
    13: 'irrelevant_signs',
    14: 'road_blocks',
    15: 'tractor',
    16: 'non-drivable_street',
    17: 'zebra_crossing',
    18: 'obstacles/trash',
    19: 'poles',
    20: 'RD_restricted_area',
    21: 'animals',
    22: 'grid_structure',
    23: 'signal_corpus',
    24: 'drivable_cobblestone',
    25: 'electronic_traffic',
    26: 'slow_drive_area',
    27: 'nature_object',
    28: 'parking_area',
    29: 'sidewalk',
    30: 'ego_car',
    31: 'painted_driv._instr.',
    32: 'traffic_guide_obj.',
    33: 'dashed_line',
    34: 'RD_normal_street',
    35: 'sky',
    36: 'buildings',
    37: 'blurred_area',
    38: 'rain_dirt'
}


cls_freq = [0, 16638586, 816746, 885671, 4205546, 166147, 209321, 
            1277733, 544559, 32109, 3, 5093660, 1705323, 2194196, 
            1044710, 5349, 3029528, 161433, 1668462, 2647306, 956223, 
            4182, 4622371, 439294, 6069454, 9990, 1138946, 78342740, 
            2156414, 21557480, 8634634, 660671, 1394186, 1719920, 85871754, 
            2745726, 63773755, 9046, 45]
            

unused_index = [942, 12124, 12125, 12126, 12127, 12128, 12129, 12130, 12131, 12132, 12133, 12134, 20720, 20721, 20722, 20723, 20724, 20725, 20726, 20727, 21299, 21300, 21301, 21302, 27427, 27428]

zero_size_index = [12907, 12908, 12909, 12910, 12911, 12912, 13649, 13650, 13651, 13652]


class A2D2_PV(data.Dataset):
    def __init__(self, root, camsLidars_path, classIndex_path, split='train', has_label=True):
        self.root = root
        self.split = split
        self.has_label = has_label
        self.mapped_class_name = mapped_class_name
        self.unused_index = unused_index

        # PMFv1
        self.cls_freq = np.array(cls_freq)
        self.cls_freq = self.cls_freq / self.cls_freq.sum()
        self.cls_freq[0] = 0

        # remove zero_size
        self.zero_size_index = zero_size_index

        with open(camsLidars_path, 'r') as f:
            self.cams_lidars = json.load(f)

        # with open(classList_path, 'r') as f:
        #     self.class_list = json.load(f)

        with open(classIndex_path, 'r') as f:
            self.class_index = json.load(f)

        if os.path.isdir(self.root):
            print("Dataset found: {}".format(self.root))
        else:
            raise ValueError("dataset not found: {}".format(self.root))
        
        self.lidar_files = []
        self.camera_files = []
        self.label_files = []

        self.lidar_files = sorted(glob.glob(os.path.join(self.root, '*/lidar/*/*.npz')))
        if self.unused_index is not None:
            self.lidar_files = np.delete(self.lidar_files, self.unused_index)

        # remove zero_size
        if self.zero_size_index is not None:
            self.lidar_files = np.delete(self.lidar_files, self.zero_size_index)

        if self.split == 'train':
            self.lidar_files = self.lidar_files[:22407]
        elif self.split == 'valid':
            self.lidar_files = self.lidar_files[22407:25181]
        elif self.split == 'test':
            self.lidar_files = self.lidar_files[25181:]
        elif self.split == 'all':
            self.lidar_files == self.lidar_files
        else:
            raise ValueError("invalid split: {}".format(self.split))
        
        self.camera_files = self.extract_file_name_from_lidar_file_name(self.lidar_files, "camera")
        self.label_files = self.extract_file_name_from_lidar_file_name(self.lidar_files, "label")

        assert len(self.lidar_files) == len(self.camera_files) and len(self.camera_files) == len(self.label_files), "Error: number of files must be the same"


    @staticmethod
    def extract_file_name_from_lidar_file_name(lidar_files, name="camera"):
        if name != "camera" and name != "label":
            raise ValueError("parameter error: {}".format(name))
        
        files = []
        if name == "camera":
            for lidar_file in lidar_files:
                file_splits = lidar_file.split('/')
                file_splits[-3] = file_splits[-3].replace('lidar', 'camera')
                file_splits[-1] = file_splits[-1].replace('lidar', 'camera')
                file_splits[-1] = file_splits[-1].replace('npz', 'png')
                file = os.path.join('/', *file_splits)
                files.append(file)
        else:
            for lidar_file in lidar_files:
                file_splits = lidar_file.split('/')
                file_splits[-3] = file_splits[-3].replace('lidar', 'label')
                file_splits[-1] = file_splits[-1].replace('lidar', 'label')
                file_splits[-1] = file_splits[-1].replace('npz', 'png')
                file = os.path.join('/', *file_splits)
                files.append(file)

        return files

    
    @staticmethod
    def get_save_file_name(file_name):
        save_file_name = file_name.split('/')[-1].replace('label', 'pred')
        save_file_name = save_file_name.replace('png', 'label')
        return save_file_name
        

    def undistort_image(self, image, cam_name):
        if cam_name in ['frontleft', 'frontcenter', \
                        'frontright', 'sideleft', \
                        'sideright', 'rearcenter']:

            if cam_name[0] == 'f':
                cam_name = cam_name[:5] + '_' + cam_name[5:]
            else:
                cam_name = cam_name[:4] + '_' + cam_name[4:]
    
            # get parameters from config file
            intr_mat_undist = \
                    np.asarray(self.cams_lidars['cameras'][cam_name]['CamMatrix'])
            intr_mat_dist = \
                    np.asarray(self.cams_lidars['cameras'][cam_name]['CamMatrixOriginal'])
            dist_parms = \
                    np.asarray(self.cams_lidars['cameras'][cam_name]['Distortion'])
            lens = self.cams_lidars['cameras'][cam_name]['Lens']

            if (lens == 'Fisheye'):
                return cv2.fisheye.undistortImage(image, intr_mat_dist,\
                                        D=dist_parms, Knew=intr_mat_undist)
            elif (lens == 'Telecam'):
                return cv2.undistort(image, intr_mat_dist, \
                        distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
            else:
                return image
        else:
            return image
  

    @staticmethod
    def loadLidarData(path):
        lidar_data = np.load(path)
        return lidar_data
    

    @staticmethod
    def loadSemImage(path):
        semantic_image = Image.open(path)
        semantic_image = np.array(semantic_image)
        return semantic_image
       

    def loadImage(self, index):
        path = self.camera_files[index]
        image = Image.open(path)
        image = np.array(image)
        camera_name = path.split('/')[-1]
        camera_name = camera_name.split('.')[0]
        camera_name = camera_name.split('_')[2]
        undist_image = self.undistort_image(image, camera_name)
        undist_image = Image.fromarray(undist_image)
        return undist_image

    
    def parsePathInfoByIndex(self, index):
        return index, ''


    def loadDataByIndex(self, index):
        lidar_data = self.loadLidarData(self.lidar_files[index])
        pointcloud = lidar_data['points']
        reflectance = lidar_data['reflectance']
        reflectance = np.expand_dims(reflectance, axis=1)
        pointcloud = np.concatenate((pointcloud, reflectance), axis=1)
        
        if self.has_label:
            semantic_image = self.loadSemImage(self.label_files[index])
            semantic_label = self.getLabel(lidar_data, semantic_image)
            instance_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        else:
            semantic_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
            instance_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        
        return pointcloud, semantic_label, instance_label


    @staticmethod
    def rgb_to_hex(color):
        strs = '#'
        for i in range(3):
            strs += str(hex(np.int32(color[i])))[-2:].replace('x', '0').lower()
        return strs

    
    def getLabel(self, lidar_data, sem_image):
        label = np.zeros(len(lidar_data['points']), dtype=np.int32)
        rows = (lidar_data['row'] + 0.5).astype(np.int32)
        cols = (lidar_data['col'] + 0.5).astype(np.int32)

        for i in range(len(lidar_data['points'])):
            row = rows[i]
            col = cols[i]
            label[i] = self.class_index[self.rgb_to_hex(sem_image[row, col])]

        return label

    
    def labelMapping(self, label):
        return label


    def mapLidar2Camera(self, index, pointcloud, img_h, img_w):
        lidar_data = self.loadLidarData(self.lidar_files[index])
        rows = (lidar_data['row'] + 0.5).astype(np.int32)
        cols = (lidar_data['col'] + 0.5).astype(np.int32)
        mapped_points = np.array(list(zip(rows, cols)))
        keep_mask = np.full(len(lidar_data['points']), True, dtype=bool)

        return mapped_points, keep_mask
        

    def mapLidar2CameraCropYaw(self, index, pointcloud):
        lidar_data = self.loadLidarData(self.lidar_files[index])
        rows = (lidar_data['row'] + 0.5).astype(np.int32)
        cols = (lidar_data['col'] + 0.5).astype(np.int32)
        mapped_points = np.array(list(zip(rows, cols)))
        mapped_points = mapped_points.reshape(-1, 2)
        keep_mask = np.full(len(lidar_data['points']), True, dtype=bool)

        return pointcloud, mapped_points, keep_mask

    
    def loadLabelByIndex(self, index):
        lidar_data = self.loadLidarData(self.lidar_files[index])
        nums = len(lidar_data['points'])
        
        if self.has_label:
            semantic_image = self.loadSemImage(self.label_files[index])
            semantic_label = self.getLabel(lidar_data, semantic_image)
            instance_label = np.zeros(nums, dtype=np.int32)
        else:
            semantic_label = np.zeros(nums, dtype=np.int32)
            instance_label = np.zeros(nums, dtype=np.int32)
        
        return semantic_label, instance_label


    def __len__(self):
        return len(self.lidar_files)
    

    def __getitem__(self, index):
        lidar_data = self.loadLidarData(self.lidar_files[index])
        undist_image = self.loadRawImage(self.camera_files[index])
        semantic_image = self.loadSemImage(self.label_files[index])
        semantic_label = self.getLabel(lidar_data, semantic_image)
        return lidar_data, undist_image, semantic_label