import os
from metrics import Stixel
from typing import List
import csv


def read_stixel_from_csv(filename):
    stixels = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row:  # avoid empty rows
                x = int(row[1])
                y_t = int(row[2])
                y_b = int(row[3])
                sem_class = int(row[4])
                depth = float(row[5])
                stixel = Stixel(x, y_t, y_b, depth, sem_class)
                stixels.append(stixel)
    file = os.path.basename(filename)
    return {"sample": os.path.splitext(file)[0], "target": stixels}


class EvaluationDataloader:
    def __init__(self, target_folder):
        self.targets = []
        self.predictions = [{}]
        for target_file in os.listdir(target_folder):
            self.targets.append(read_stixel_from_csv(target_file))

    def __getitem__(self, idx):
        for i in range(len(self.targets)):
            if self.predictions[idx]["sample"] == self.targets[i]["sample"]:
                return self.predictions[idx]["prediction"], self.predictions[i]["target"]
        sample_name = self.predictions[idx]["sample"]
        print(f"Haven't found any target for prediction: {sample_name}.")
