import os
from dataloader.EvaluationDataloader import EvaluationDataloader, read_stixel_from_csv


class StixelNExTLoader(EvaluationDataloader):
    def __init__(self, prediction_file, target_folder):
        super().__init__(target_folder)
        for target_file in os.listdir(target_folder):
            self.predictions.append(read_stixel_from_csv(target_file))
        if len(self.predictions) != len(self.targets):
            print(f"INFO: Inconsistent number of predictions[{len(self.predictions)}] and targets[{len(self.targets)}]")

    def __len__(self):
        return len(self.predictions)
