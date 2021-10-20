
import sys 
sys.path.insert(0, "../../")

from pc_processor.dataset.semantic_kitti import SemanticKitti
import numpy as np
import os
import shutil

def createFovDataset(src_root, dst_root, seq):
    dataset = SemanticKitti(
        root=src_root,
        sequences=[seq],
        config_path="../../pc_processor/dataset/semantic_kitti/semantic-kitti.yaml"
    )
    
    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)

    data_len = len(dataset)
    for i in range(data_len):
        print("processing {}|{} ...".format(data_len,i))
        pointcloud, sem_label, inst_label = dataset.loadDataByIndex(i)
        image = dataset.loadImage(i)

        image = np.array(image)
        seq_id, frame_id = dataset.parsePathInfoByIndex(i)
        mapped_pointcloud, keep_mask = dataset.mapLidar2Camera(
            seq_id, pointcloud[:, :3], image.shape[1], image.shape[0])
        
        keep_pointcloud = pointcloud[keep_mask]
        keep_sem_label = sem_label[keep_mask].astype(np.int32)
        keep_inst_label = inst_label[keep_mask].astype(np.int32)
        keep_label = (keep_inst_label << 16) + keep_sem_label

        # check path
        pointcloud_path = os.path.join(dst_root, seq_id, "velodyne")
        if not os.path.isdir(pointcloud_path):
            os.makedirs(pointcloud_path)

        label_path = os.path.join(dst_root, seq_id,  "labels")
        if not os.path.isdir(label_path):
            os.makedirs(label_path)
        
        pointcloud_file = os.path.join(pointcloud_path, "{}.bin".format(frame_id))
        label_file = os.path.join(label_path, "{}.label".format(frame_id))
        keep_pointcloud.tofile(pointcloud_file)
        keep_label.tofile(label_file)
    
    print("copy image_2 folder ...")
    # copy image and calib files
    src_img_folder = os.path.join(src_root, "{:02d}".format(seq), "image_2")
    dst_img_folder = os.path.join(dst_root, "{:02d}".format(seq), "image_2")
    shutil.copytree(src_img_folder, dst_img_folder)

    target_files = ["calib.txt", "poses.txt", "times.txt"]
    print("copy calib files ...")
    for f_name in target_files:
        src_file_path = os.path.join(src_root, "{:02d}".format(seq), f_name)
        dst_file_path = os.path.join(dst_root, "{:02d}".format(seq), f_name)
        shutil.copyfile(src_file_path, dst_file_path)


"""
extract fov data from semantic-kitti and construct data set semantic-kitti-fov
"""
if __name__ == "__main__":
    for seq in range(0, 11):
        createFovDataset(
            src_root="/path/to/semantic-kitti/sequences", # path to the original semantic-kitti dataset
            dst_root="/path/to/semantic-kitti-fov/sequences", # path to the generated semantic-kitti-fov dataset
            seq=seq
        )