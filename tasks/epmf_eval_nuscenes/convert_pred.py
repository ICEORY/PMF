import os
import numpy as np

root_folder = "/home/iceory/project/PMFv2/experiments/PMF-nuScenes/log_nuScenes_PMFNetV2-resnet34_bs24-lr0.001_E150-MTL-ASPP-2022121301/Eval-nuScenes-PMFNet-best_IOU_model-noKNN-test_20221227"
pred_folder = os.path.join(root_folder, "preds", "lidarseg", "test")
filename_list = os.listdir(pred_folder)
for i, filename in enumerate(filename_list):
    if ".bin" in filename:
        file_path = os.path.join(pred_folder, filename)
        pred = np.fromfile(file_path, dtype=np.int32)
        pred = pred.astype(np.uint8)
        # print(file_path)
        # assert False
        pred.tofile(file_path)
        print("[{}] converting ...".format(i))