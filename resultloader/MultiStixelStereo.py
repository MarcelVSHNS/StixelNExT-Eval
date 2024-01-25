import os
from resultloader.EvaluationDataloader import EvaluationDataloader, read_stixel_from_csv
import yaml
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
from metrics.ObstacleDetection import get_obstacles_only
import cv2
import numpy as np


class StereoStixelLoader(EvaluationDataloader):
    def __init__(self, obstacle_detection_mode=False):
        if obstacle_detection_mode:
            dataset_snippet = 'cityscapes/kitti'
        else:
            dataset_snippet = config['dataset']
        self.prediction_folder = os.path.join(config['data_path'], dataset_snippet, 'testing', 'predictions_from_StixelNet')
        super().__init__(os.path.join(os.path.dirname(self.prediction_folder), "targets_from_lidar"))
        self.prediction_list = [pred for pred in os.listdir(self.prediction_folder) if pred.endswith(".csv")]
        self.image_folder = os.path.join(os.path.dirname(self.prediction_folder), "STEREO_LEFT")
        self.obstacle_detection_mode = obstacle_detection_mode
        if len(self.prediction_list) != len(self.target_list):
            print(f"INFO: Inconsistent number of predictions[{len(self.prediction_list)}] and targets[{len(self.target_list)}]:")
            print("In target but not in predictions")
            for name in self.target_list:
                if name not in self.prediction_list:
                    print(name)
            print("In prediction but not in targets")
            for name in self.prediction_list:
                if name not in self.target_list:
                    print(name)
        else:
            print(f"Found {len(self.prediction_list)} samples.")

    def __getitem__(self, idx):
        filename = self.prediction_list[idx]
        pred_stx = read_stixel_from_csv(os.path.join(self.prediction_folder, filename))
        targ_stx = read_stixel_from_csv(os.path.join(self.target_folder, filename))
        if self.obstacle_detection_mode:
            img_filename = os.path.splitext(filename)[0] + ".png"
            image = cv2.imread(os.path.join(self.image_folder, img_filename))
            pred_stx = get_obstacles_only(pred_stx)
            return pred_stx, targ_stx, image
        else:
            return pred_stx, targ_stx

    def __len__(self):
        return len(self.prediction_list)
