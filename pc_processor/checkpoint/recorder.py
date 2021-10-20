import os
import shutil
import logging
import sys
import tensorboardX

class Recorder(object):
    def __init__(self, settings, save_path, use_tensorboard=True):
        print('>> Init a recoder at ', save_path)
        self.save_path = save_path
        self.settings = settings
        self.code_path = os.path.join(self.save_path, "code")
        self.log_path = os.path.join(self.save_path, "log")
        self.code_file_extension = [".py", ".yml", ".yaml", ".sh"]
        self.ignore_file_extension = [".pyc"]
        self.checkpoint_path = os.path.join(self.save_path, "checkpoint")
        if use_tensorboard:
            self.tensorboard = tensorboardX.SummaryWriter(logdir=self.save_path)
        else:
            self.tensorboard = None

        # make directories
        if not os.path.isdir(self.code_path):
            os.makedirs(self.code_path)
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.isdir(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        # init logger
        self.logger = self._initLogger()
        # save code and settings
        self._saveConfig()

    def _initLogger(self):
        logger = logging.getLogger("console")
        logger.propagate = False
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        console_formatter = logging.Formatter('%(message)s')
        # file log
        file_handler = logging.FileHandler(os.path.join(self.log_path, "console.log"))
        file_handler.setFormatter(file_formatter)

        # console log
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)
        return logger

    def _checkFileExtension(self, file_name):
        for k in self.code_file_extension:
            if k in file_name:
                for kk in self.ignore_file_extension:
                    if kk in file_name:
                        return False
                return True
        return False

    def _copyFiles(self, root_path, target_path):
        file_list = os.listdir(root_path)
        for file_name in file_list:
            file_path = os.path.join(root_path, file_name)
            if os.path.isdir(file_path) and "log_" not in file_path:
                dst_path = os.path.join(target_path, file_path)
                self._copyFiles(file_path, dst_path)
            else:
                if self._checkFileExtension(file_name):
                    if not os.path.isdir(target_path):
                        os.makedirs(target_path)
                    dst_file = os.path.join(target_path, file_name)
                    shutil.copyfile(file_path, dst_file)
                
    def _saveConfig(self):
        # copy code files
        self._copyFiles(root_path="./", target_path=self.code_path)

        # write settings to file
        with open(os.path.join(self.log_path, "settings.log"), "w") as f:
            for k, v in self.settings.__dict__.items():
                f.write("{}: {}\n".format(k, v))