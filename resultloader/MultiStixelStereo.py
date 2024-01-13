import os
from resultloader import EvaluationDataloader, read_stixel_from_csv


class StereoStixelLoader(EvaluationDataloader):
    def __init__(self, prediction_folder):
        self.prediction_folder = prediction_folder
        base_path = prediction_folder.split("/")[:-1]
        super().__init__(os.path.join("/".join(base_path), "targets_from_lidar"))
        self.prediction_list = [pred for pred in os.listdir(self.prediction_folder) if pred.endswith(".csv")]
        if len(self.prediction_list) != len(self.target_list):
            print(f"INFO: Inconsistent number of predictions[{len(self.prediction_list)}] and targets[{len(self.target_list)}]")

    def __getitem__(self, idx):
        filename = self.prediction_list[idx]
        pred_stx = read_stixel_from_csv(os.path.join(self.prediction_folder, filename))
        targ_stx = read_stixel_from_csv(os.path.join(self.target_folder, filename))
        return pred_stx, targ_stx

    def __len__(self):
        return len(self.prediction_list)
