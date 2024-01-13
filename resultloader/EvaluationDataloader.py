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
    def __init__(self, target_folder: str):
        self.target_folder = target_folder
        self.target_list = [targ for targ in os.listdir(self.target_folder) if targ.endswith(".csv")]

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.target_list)
