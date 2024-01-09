import wandb
import yaml
import os
import numpy as np
from metrics.PrecisionRecall import PrecisionRecall
from dataloader import StixelNExTLoader as Dataloader

with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def main():
    test_dataloader = Dataloader(prediction_file=config['result_file'],
                                 target_folder=config['test_files'])

    # Create an export of analysed data
    if config['logging']['activate']:
        # Init the logger
        # e.g. StixelNExT_ancient-silence-25_epoch-94_loss-0.09816327691078186.pth
        run = os.path.splitext(os.path.basename(config['result_file']))[0]
        epochs = run.split('_')[2].split('-')[1]
        checkpoint = run.split('_')[1]
        dataset = os.path.basename(config['test_files'])
        wandb_logger = wandb.init(project=config['logging']['project'],
                                  config={
                                      "run": run.split('_')[1],
                                      "dataset": os.path.basename(config['test_files']),
                                      "checkpoint": run,
                                      "epochs": run.split('_')[2].split('-')[1]
                                  },
                                  tags=["metrics", "testing"]
                                  )

        thresholds = np.linspace(0.1, 1.0, num=config['num_thresholds'])
        # A list with all average prec. and recall by IoU
        precision_by_iou = []
        recall_by_iou = []
        f1_score_by_iou = []
        for iou in thresholds:
            metric = PrecisionRecall(iou_threshold=iou)
            # A list with precision and recall per sample of testdata
            precision_by_testdata = []
            recall_by_testdata = []
            f1_score_by_testdata = []
            for predictions, targets in test_dataloader:
                # iterate over all samples
                precision, recall = metric.evaluate(predictions, targets)
                precision_by_testdata.append(precision)
                recall_by_testdata.append(recall)
                f1_score = metric.get_score()
                f1_score_by_testdata.append(f1_score)
            # calculate average over all samples and append to IoU list
            precision_by_iou.append(sum(precision_by_testdata) / len(precision_by_testdata))
            recall_by_iou.append(sum(recall_by_testdata) / len(recall_by_testdata))
            f1_score_by_iou.append(sum(f1_score_by_testdata) / len(f1_score_by_testdata))

        data = [[x, y] for (x, y) in zip(recall_by_iou, precision_by_iou)]
        table = wandb.Table(data=data, columns=["Recall", "Precision"])
        wandb_logger.log({"PR curve": wandb.plot.line(table, "Recall", "Precision",
                                                      title=f"Precision-Recall over {len(test_dataloader)} samples")})
        wandb_logger.log({"F1": sum(f1_score_by_iou) / len(f1_score_by_iou)})


if __name__ == '__main__':
    main()
