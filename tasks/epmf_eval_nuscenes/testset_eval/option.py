import os 
import yaml
import sys 
import shutil

sys.path.insert(0, "../../../")

import pc_processor

class Option(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, "r"))

        # ---------------------------- general options -----------------
        self.save_path = self.config["save_path"] # log path
        self.experiment_id = self.config["experiment_id"]
        self.is_debug = self.config["is_debug"]
        self.dataset = self.config["dataset"]
        self.data_root = self.config["data_root"]
        self.n_classes = self.config["n_classes"]
        self.has_label = self.config["has_label"]
        self.pred_folder = self.config["pred_folder"]
        self._prepare()

    def _prepare(self):
        # ---- check params
        self.save_path = os.path.join(self.config["save_path"], self.experiment_id)
        if not os.path.isdir(self.pred_folder):
            raise FileNotFoundError("main prediction folder not found: {}".format(self.pred_folder))
        self.check_path()


    def check_path(self): 
        if pc_processor.utils.is_main_process():
            if os.path.exists(self.save_path):
                # shutil.rmtree(self.save_path)
                print("file exist: {}".format(self.save_path))
                action = input("Select Action: d(delete) / q(quit): ").lower().strip()
                if action == "d":
                    shutil.rmtree(self.save_path)
                else:
                    raise OSError("Directory exits: {}".format(self.save_path))
        
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
