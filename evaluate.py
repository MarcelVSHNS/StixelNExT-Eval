import wandb
import yaml
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
import os
from metrics.PrecisionRecall import PrecisionRecall
from datetime import datetime
from resultloader import StereoStixelLoader as Dataloader

overall_start_time = datetime.now()


def main():
    test_data_generator = Dataloader()
    metric = PrecisionRecall(iou_threshold=config['evaluation']['iou'], rm_used=False)
    # Create an export of analysed data
    if config['logging']['activate']:
        run_logger = "Stereo"
        checkpoint = "-"
        epochs = "-"

        wandb_logger = wandb.init(project=config['logging']['project'],
                                  config={
                                      "run": run_logger,
                                      "dataset": config['dataset'],
                                      "checkpoint": checkpoint,
                                      "epochs": epochs
                                  },
                                  tags=["metrics", "testing"]
                                  )

        # A list with precision and recall per sample of testdata
        precision_by_testdata = []
        recall_by_testdata = []
        f1_score_by_testdata = []
        for sample in test_data_generator:
            try:
                prediction, target = sample
            except FileNotFoundError as e:
                continue
            # iterate over all samples
            precision, recall = metric.evaluate(prediction, target)
            precision_by_testdata.append(precision)
            recall_by_testdata.append(recall)
            f1_score = metric.get_score()
            f1_score_by_testdata.append(f1_score)
        avg_prec = sum(precision_by_testdata) / len(precision_by_testdata)
        avg_rec = sum(recall_by_testdata) / len(recall_by_testdata)
        avg_f1 = sum(f1_score_by_testdata) / len(f1_score_by_testdata)
        print(f"... Precision: {avg_prec}")
        print(f"... Recall: {avg_rec}")
        print(f"... F1 Score: {avg_f1}")
        # calculate average over all samples and append to IoU list
        step_time = datetime.now() - overall_start_time
        print("Time elapsed: {}".format(step_time))

        wandb_logger.log({"Precision": avg_prec})
        wandb_logger.log({"Recall": avg_rec})
        wandb_logger.log({"F1": avg_f1})


if __name__ == '__main__':
    main()
