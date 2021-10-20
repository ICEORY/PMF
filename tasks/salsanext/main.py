import argparse
import datetime
from option import Option
import os
import torch
from trainer import Trainer
import pc_processor
import time


class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings

        # init gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = self.settings.gpu
        pc_processor.utils.init_distributed_mode(self.settings)
        torch.distributed.barrier()

        self.settings.check_path()
        # set random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        torch.cuda.set_device(self.settings.gpu)
        torch.backends.cudnn.benchmark = True

        # init checkpoint
        if not self.settings.distributed or (self.settings.rank == 0):
            self.recorder = pc_processor.checkpoint.Recorder(self.settings, self.settings.save_path)
        else:
            self.recorder = None

        self.epoch_start = 0
        # init model
        self.model = self._initModel()

        # init trainer
        self.trainer = Trainer(self.settings, self.model, self.recorder)

        # load checkpoint
        self._loadCheckpoint()


    def _initModel(self):
        if self.settings.net_type == "SalsaNext":
            model = pc_processor.models.SalsaNext(in_channels=5, nclasses=self.settings.n_classes)
        else:
            raise ValueError("invalid model: {}".format(
                self.settings.net_type))
        return model

    def _loadCheckpoint(self):
        assert self.settings.pretrained_model is None or self.settings.checkpoint is None, "cannot use pretrained weight and checkpoint at the same time"
        if self.settings.pretrained_model is not None:
            if not os.path.isfile(self.settings.pretrained_model):
                raise FileNotFoundError("pretrained model not found: {}".format(self.settings.pretrained_model))
            state_dict = torch.load(self.settings.pretrained_model)
            self.model.load_state_dict(state_dict)
            if self.recorder is not None:
                self.recorder.logger.info("loading pretrained weight from: {}".format(self.settings.pretrained_model))

        if self.settings.checkpoint is not None:
            if not os.path.isfile(self.settings.checkpoint):
                raise FileNotFoundError("checkpoint file not found: {}".format(self.settings.checkpoint))
            checkpoint_data = torch.load(self.settings.checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint_data["model"])
            self.trainer.optimizer.load_state_dict(checkpoint_data["optimizer"])
            self.epoch_start = checkpoint_data["epoch"] + 1

    def run(self):
        t_start = time.time()
        if self.settings.val_only:
            self.trainer.run(0, mode="Validation")
            return
        best_val_result = None
        self.trainer.scheduler.step(self.epoch_start*len(self.trainer.train_loader))
        
        for epoch in range(self.epoch_start, self.settings.n_epochs):
            self.trainer.run(epoch, mode="Train")
            if epoch % self.settings.val_frequency == 0 or epoch == self.settings.n_epochs - 1:
                val_result = self.trainer.run(epoch, mode="Validation")

                # save best result
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
                    "model": self.model.state_dict(),  # .cpu().state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(checkpoint_data, saved_path)
                # log
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
