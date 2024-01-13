import wandb
import yaml
import os
import numpy as np
from metrics.PrecisionRecall import PrecisionRecall
from resultloader import StixelNExTLoader as Dataloader
from datetime import datetime

with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
overall_start_time = datetime.now()


def main():
    test_dataloader = Dataloader()

    # Create an export of analysed data
    if config['logging']['activate']:
        # Init the logger
        # e.g. StixelNExT_ancient-silence-25_epoch-94_loss-0.09816327691078186.pth predictions_from_stereo
        run = os.path.basename(config['prediction_folder'])
        if config['prediction_folder'].split("/")[-1] == "predictions_from_stereo":
            run = "Stereo"
            checkpoint = "-"
            epochs = "-"
        else:
            run_logger = run.split('_')[1]
            checkpoint = run
            epochs = run.split('_')[2].split('-')[1]

        wandb_logger = wandb.init(project=config['logging']['project'],
                                  config={
                                      "run": run_logger,
                                      "dataset": config['dataset'],
                                      "checkpoint": checkpoint,
                                      "epochs": epochs
                                  },
                                  tags=["metrics", "testing"]
                                  )

        thresholds = np.linspace(0.1, 1.0, num=config['num_thresholds'])
        # A list with all average prec. and recall by IoU
        precision_by_iou = []
        recall_by_iou = []
        f1_score_by_iou = []
        for iou in thresholds:
            print(f"Evaluating threshold: {iou} ...")
            metric = PrecisionRecall(iou_threshold=iou)
            # A list with precision and recall per sample of testdata
            precision_by_testdata = []
            recall_by_testdata = []
            f1_score_by_testdata = []
            for i in range(len(test_dataloader)):
                try:
                    prediction, target = test_dataloader[i]
                except FileNotFoundError as e:
                    continue
                # iterate over all samples
                precision, recall = metric.evaluate(prediction, target)
                precision_by_testdata.append(precision)
                recall_by_testdata.append(recall)
                f1_score = metric.get_score()
                f1_score_by_testdata.append(f1_score)
            iou_avg_prec = sum(precision_by_testdata) / len(precision_by_testdata)
            iou_avg_rec = sum(recall_by_testdata) / len(recall_by_testdata)
            iou_avg_f1 = sum(f1_score_by_testdata) / len(f1_score_by_testdata)
            print(f"... Precision: {iou_avg_prec}")
            print(f"... Recall: {iou_avg_rec}")
            print(f"... F1 Score: {iou_avg_f1}")
            # calculate average over all samples and append to IoU list
            precision_by_iou.append(iou_avg_prec)
            recall_by_iou.append(iou_avg_rec)
            f1_score_by_iou.append(iou_avg_f1)
            step_time = datetime.now() - overall_start_time
            print("Time elapsed: {}".format(step_time))

        data = [[x, y] for (x, y) in zip(recall_by_iou, precision_by_iou)]
        table = wandb.Table(data=data, columns=["Recall", "Precision"])
        wandb_logger.log({"PR curve": wandb.plot.line(table, "Recall", "Precision",
                                                      title=f"Precision-Recall over {len(test_dataloader)} samples")})
        wandb_logger.log({"F1": sum(f1_score_by_iou) / len(f1_score_by_iou)})


if __name__ == '__main__':
    main()
