import argparse
import datetime
from option import Option
import os
import torch
import time
import trainer

import pc_processor

class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings
        # init gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = self.settings.gpu
        pc_processor.utils.init_distributed_mode(self.settings)
        torch.distributed.barrier()

        # set random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        torch.cuda.set_device(self.settings.gpu)
        torch.backends.cudnn.benchmark = True

        # init checkpoint
        if not self.settings.distributed or (self.settings.rank == 0):
            self.recorder = pc_processor.checkpoint.Recorder(
                self.settings, self.settings.save_path)
        else:
            self.recorder = None

        self.epoch_start = 0
        # init model
        self.model = pc_processor.models.PMFNet(
            pcd_channels=5,
            img_channels=3,
            nclasses=self.settings.nclasses,
            base_channels=self.settings.base_channels,
            image_backbone=self.settings.img_backbone,
            imagenet_pretrained=self.settings.imagenet_pretrained
        )

        # init trainer
        self.trainer = trainer.Trainer(
            self.settings, self.model, self.recorder)
        # load checkpoint
        self._loadCheckpoint()

    def _loadCheckpoint(self):
        assert self.settings.pretrained_model is None or self.settings.checkpoint is None, "cannot use pretrained weight and checkpoint at the same time"
        if self.settings.pretrained_model is not None:
            if not os.path.isfile(self.settings.pretrained_model):
                raise FileNotFoundError("pretrained model not found: {}".format(
                    self.settings.pretrained_model))
            state_dict = torch.load(
                self.settings.pretrained_model, map_location="cpu")
            new_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                if k in new_state_dict.keys():
                    if new_state_dict[k].size() == v.size():
                        new_state_dict[k] = v
                    else:
                        print("diff size: ", k, v.size())
                else:
                    print("diff key: ", k)
            self.model.load_state_dict(new_state_dict)
            # self.model.load_state_dict(state_dict)
            if self.recorder is not None:
                self.recorder.logger.info(
                    "loading pretrained weight from: {}".format(self.settings.pretrained_model))

        if self.settings.checkpoint is not None:
            if not os.path.isfile(self.settings.checkpoint):
                raise FileNotFoundError(
                    "checkpoint file not found: {}".format(self.settings.checkpoint))
            checkpoint_data = torch.load(
                self.settings.checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint_data["model"])
            self.trainer.optimizer.load_state_dict(
                checkpoint_data["optimizer"])
            self.trainer.aux_optimizer.load_state_dict(
                checkpoint_data["aux_optimizer"])
            self.epoch_start = checkpoint_data["epoch"] + 1

    def run(self):
        t_start = time.time()
        if self.settings.val_only:
            self.trainer.run(0, mode="Validation")
            return
        
        best_val_result = None
        # self.epoch_start = 1
        # update lr after backward (required by pytorch)
        # self.trainer.scheduler.step(self.epoch_start*len(self.trainer.train_loader))
        # if self.settings.optimizer == "Hybrid":
        #     self.trainer.aux_scheduler.step(self.epoch_start*len(self.trainer.train_loader))

        for epoch in range(self.epoch_start, self.settings.n_epochs):
            
            self.trainer.run(epoch, mode="Train")
            if epoch % self.settings.val_frequency == 0 or epoch == self.settings.n_epochs-1:
                val_result = self.trainer.run(epoch, mode="Validation")

                if self.recorder is not None:
                    if best_val_result is None:
                        best_val_result = val_result
                    for k, v in val_result.items():
                        if v >= best_val_result[k]:
                            self.recorder.logger.info(
                                "get better {} model: {}".format(k, v))
                            saved_path = os.path.join(
                                self.recorder.checkpoint_path, "best_{}_model.pth".format(k))
                            best_val_result[k] = v
                            torch.save(self.model.state_dict(), saved_path)

            # save checkpoint
            if self.recorder is not None:
                saved_path = os.path.join(
                    self.recorder.checkpoint_path, "checkpoint.pth")
                checkpoint_data = {
                    "model": self.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "aux_optimizer": self.trainer.aux_optimizer.state_dict(),
                    "epoch": epoch,
                }

                torch.save(checkpoint_data, saved_path)
                # log
                if best_val_result is not None:
                    log_str = ">>> Best Result: "
                    for k, v in best_val_result.items():
                        log_str += "{}: {} ".format(k, v)
                    self.recorder.logger.info(log_str)
        cost_time = time.time() - t_start
        if self.recorder is not None:
            self.recorder.logger.info("==== total cost time: {}".format(
                datetime.timedelta(seconds=cost_time)))


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
