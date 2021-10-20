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
        self.save_path = self.config["training_folder"] # log path
        self.seed = self.config["seed"] # manually set RNG seed
        self.gpu = self.config["gpu"] # GPU id to use, e.g. "0,1,2,3"
        self.rank = 0 # rank of distributed thread
        self.world_size = 1 # 
        self.distributed = False # 
        self.n_gpus = len(self.gpu.split(",")) # # number of GPUs to use by default
        self.dist_backend = "nccl"
        self.dist_url = "env://"

        self.print_frequency = self.config["print_frequency"]  # print frequency (default: 10)
        self.n_threads = self.config["n_threads"] # number of threads used for data loading
        self.experiment_id = self.config["experiment_id"] # identifier for experiment
        self.is_debug = self.config["is_debug"]
       
        # --------------------------- data config ------------------------
        self.dataset = self.config["dataset"]
        self.nclasses = self.config["n_classes"]
        self.data_root = self.config["data_root"]
        self.has_label = self.config["has_label"]

        # --------------------------- model options -----------------------
        self.img_backbone = self.config["img_backbone"]
        self.base_channels = self.config["base_channels"]
        self.imagenet_pretrained = self.config["imagenet_pretrained"]
        
        # ---- pdff net config
        self.feature_mean = self.config["feature_mean"]
        self.feature_std = self.config["feature_std"]
        self.img_size = self.config["img_size"]

        # --------------------------- checkpoit model ----------------------
        self.pretrained_model = os.path.join(self.config["training_folder"], "checkpoint", self.config["pretrained_model"])

        self._prepare()

    def _prepare(self):
        # ---- check params
        # --------------------------------
        if not os.path.isdir(self.save_path):
            raise ValueError("training path not exists: {}".format(self.save_path))
        
        self.save_path = os.path.join(self.save_path, "Eval-PMFNet_{}_{}".format(
            self.dataset, self.experiment_id
        ))

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
