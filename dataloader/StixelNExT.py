from dataloader.EvaluationDataloader import EvaluationDataloader
import pickle


class StixelNExTLoader(EvaluationDataloader):
    def __init__(self, prediction_file, target_folder):
        super().__init__(target_folder)
        with open(prediction_file, 'rb') as file:
            self.predictions = pickle.load(file)
        if len(self.predictions) != len(self.targets):
            print(f"INFO: Inconsistent number of predictions[{len(self.predictions)}] and targets[{len(self.targets)}]")

    def __getitem__(self, idx):
        return self.targets[idx], self.predictions[idx]

    def __len__(self):
        return len(self.predictions)
