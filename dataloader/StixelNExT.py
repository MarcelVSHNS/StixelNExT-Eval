from dataloader.EvaluationDataloader import EvaluationDataloader
import pickle
import os


class StixelNExTLoader(EvaluationDataloader):
    def __init__(self, prediction_folder):
        base_path = prediction_folder.split("/")[:-2]
        # path = os.path.join("/".join(base_path), "targets_from_lidar")
        super().__init__(os.path.join("/".join(base_path), "targets_from_lidar"))
        with open(prediction_folder, 'rb') as file:
            self.predictions = pickle.load(file)
        print("Predictions loaded.")
        if len(self.predictions) != len(self.targets):
            print(f"INFO: Inconsistent number of predictions[{len(self.predictions)}] and targets[{len(self.targets)}]")

    def __len__(self):
        return len(self.predictions)
