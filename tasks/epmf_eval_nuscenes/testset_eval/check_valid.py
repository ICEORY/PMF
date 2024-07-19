from option import Option
import argparse
import json
import os
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
        
        self.submission_json = {
            "meta": {
                "use_camera":  True,
                "use_lidar":  True,
                "use_radar":  False,
                "use_map":   False,
                "use_external": False
            },
        }

    def run(self):
        if self.settings.has_label: 
            eval_set = "val"
        else:
            eval_set = "test"
        
        submission_json_path = os.path.join(self.settings.pred_folder, "preds", eval_set)
        if not os.path.isdir(submission_json_path):
            os.makedirs(submission_json_path)
        with open(os.path.join(submission_json_path, "submission.json"), "w") as f:
            json.dump(self.submission_json, f, ensure_ascii=False, indent=4)

        validate_submission(self.nusc, eval_set=eval_set, verbose=True,
            results_folder=os.path.join(self.settings.pred_folder, "preds"), zip_out=self.settings.save_path)
        
        if self.settings.has_label:
            eval = LidarSegEval(self.nusc, eval_set=eval_set, verbose=True,
                results_folder=os.path.join(self.settings.pred_folder, "preds"))
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