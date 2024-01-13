import os
from metrics import Stixel
from typing import List
import pandas as pd


def read_stixel_from_csv(filename):
    stixels = []
    try:
        df = pd.read_csv(filename, header=0)
        for _, row in df.iterrows():
            x = int(row['x'])
            y_t = int(row['yT'])
            y_b = int(row['yB'])
            sem_class = int(row['class'])
            depth = float(row['depth'])
            stixel = Stixel(x, y_t, y_b, depth, sem_class)
            stixels.append(stixel)
        return stixels
    except FileNotFoundError as e:
        print(f"INFO:", e)
        raise e


class EvaluationDataloader:
    def __init__(self, prediction_folder):
        self.prediction_folder = prediction_folder
        if prediction_folder.split("/")[-1] == "predictions_from_stereo":
            base_path = prediction_folder.split("/")[:-1]
        else:
            base_path = prediction_folder.split("/")[:-2]
        self.target_folder = os.path.join("/".join(base_path), "targets_from_lidar")

        self.prediction_list = [pred for pred in os.listdir(self.prediction_folder) if pred.endswith(".csv")]
        self.target_list = [targ for targ in os.listdir(self.target_folder) if targ.endswith(".csv")]
        if len(self.prediction_list) != len(self.target_list):
            print(f"INFO: Inconsistent number of predictions[{len(self.prediction_list)}] and targets[{len(self.target_list)}]")

    def __getitem__(self, idx):
        filename = self.prediction_list[idx]
        pred_stx = read_stixel_from_csv(os.path.join(self.prediction_folder, filename))
        targ_stx = read_stixel_from_csv(os.path.join(self.target_folder, filename))
        return pred_stx, targ_stx

    def __len__(self):
        return len(self.prediction_list)
