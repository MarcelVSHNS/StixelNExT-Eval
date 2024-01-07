import yaml
from typing import List

with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


class Stixel:
    def __init__(self, x: int, y_t: int, y_b: int, depth: float = 8.0, sem_class: int = -1):
        self.column: int = x
        self.top: int = y_t
        self.bottom: int = y_b
        self.depth: float = depth
        self.semantic_class: int = sem_class

        self.grid_step: int = 8

    def __repr__(self):
        return f"{self.column},{self.top},{self.bottom},{self.depth}"

    def scale_by_grid(self):
        self.column = self.column * self.grid_step
        self.top = self.top * self.grid_step
        self.bottom = self.bottom * self.grid_step

    def check_integrity(self):
        # Optional: add checks like bottom is always higher than top, x-y-coordinate system, ...
        pass


class EvaluationMetric:
    def __init__(self):
        pass

    def evaluate(self, prediction, target):
        """
        apply here metrics to all predictions and targets
        :return:
            a tuple of two floats for metric
        """
        raise NotImplementedError("Please implement the -evaluate() method")

    def get_score(self):
        """
        calculate the score of a sample
        :return:
            a single float with score
        """
        raise NotImplementedError("Please implement the -get_score() method")
