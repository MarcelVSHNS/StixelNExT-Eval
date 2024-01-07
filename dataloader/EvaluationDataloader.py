import os
from metrics import Stixel
from typing import List
import csv


def read_stixel_from_csv(filename) -> List[Stixel]:
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
    return stixels


class EvaluationDataloader:
    def __init__(self, target_folder):
        self.targets = []
        for target_file in os.listdir(target_folder):
            self.targets.append(read_stixel_from_csv(target_file))

    def __getitem__(self, idx):
        """
        A generator to iterate over the prediction and target pairs
        :param idx: select data pair
        :return:
            a tuple of predictions and targets (lists with stixel-lists)
        """
        raise NotImplementedError("Please implement the __getitem__() method")
