import os
from resultloader.EvaluationDataloader import EvaluationDataloader, read_stixel_from_csv
import yaml
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


class StereoStixelLoader(EvaluationDataloader):
    def __init__(self):
        self.prediction_folder = os.path.join(config['data_path'], config['dataset'], 'testing', 'calculations_from_Stereo')
        super().__init__(os.path.join(os.path.dirname(self.prediction_folder), "targets_from_lidar"))
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
