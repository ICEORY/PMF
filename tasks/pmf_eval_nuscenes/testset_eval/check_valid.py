
import torch
from option import Option
import argparse
import datetime
import numpy as np
import os
import time
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.lidarseg.validate_submission import validate_submission
from nuscenes.eval.lidarseg.evaluate import LidarSegEval

class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings
        if self.settings.has_label:
            version = "v1.0-trainval"
        else:
            version = "v1.0-test"
        self.nusc = NuScenes(
            version=version, dataroot=self.settings.data_root, verbose=False)
        

    def run(self):
        if self.settings.has_label: 
            eval_set = "val"
        else:
            eval_set = "test"
        validate_submission(self.nusc, eval_set=eval_set, verbose=True,
            results_folder=os.path.join(self.settings.save_path, "preds"), zip_out=self.settings.save_path)
        
        if self.settings.has_label:
            eval = LidarSegEval(self.nusc, eval_set=eval_set, verbose=True,
                results_folder=os.path.join(self.settings.save_path, "preds"))
            eval.evaluate()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Options")
    parser.add_argument("config_path", type=str, metavar="config_path",
                        help="path of config file, type: string")
    parser.add_argument("--id", type=int, metavar="experiment_id", required=False,
                        help="id of experiment", default=0)
    args = parser.parse_args()
    exp = Experiment(Option(args.config_path))
    print("===init env success===")
    exp.run()