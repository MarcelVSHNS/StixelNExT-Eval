import wandb
import yaml
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
import os
from metrics.PrecisionRecall import PrecisionRecall
if config['evaluation']['model'] == "StixelNExT":
    from resultloader import StixelNExTLoader as Dataloader


def main():
    test_data_generator = Dataloader(obstacle_detection_mode=True)
    metric = PrecisionRecall(iou_threshold=config['evaluation']['iou'], rm_used=True)
    pred, targ = test_data_generator[4]


if __name__ == '__main__':
    main()
