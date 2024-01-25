import wandb
import yaml
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
import os
import numpy as np
from metrics.PrecisionRecall import PrecisionRecall
from datetime import datetime
from resultloader import StixelNExTLoader as Dataloader

overall_start_time = datetime.now()


def main():
    test_data_generator = Dataloader(exploring=config['explore_data'])
    metric = PrecisionRecall(iou_threshold=config['evaluation']['iou'], rm_used=True)
    if config['explore_data']:
        test_data_generator.set_threshold(0.8)
        pred, targ, image = test_data_generator[4]
        precision, recall = metric.evaluate(pred, targ)
        print(f"pred_len: {len(pred)}, targ_len: {len(targ)}")
        print(f'Precision: {precision}, Recall: {recall}')
        test_data_generator.stixel_reader.show_stixel(image, stixel_list=targ, color=[0, 255, 0])
        test_data_generator.stixel_reader.show_stixel(image, stixel_list=pred, color=[255, 0, 0])
    # Create an export of analysed data
    if config['logging']['activate']:
        # Init the logger
        # e.g. StixelNExT_ancient-silence-25_epoch-94_loss-0.09816327691078186.pth predictions_from_stereo
        run = config['weights_file']
        if config['evaluation']['model'] == "Stereo":
            run = "Stereo"
            checkpoint = "-"
            epochs = "-"
        else:
            run_logger = run.split('_')[1]
            checkpoint, _ = os.path.splitext(run)
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

        # overwrite of pred_threshold
        thresholds = np.linspace(0.1, 1.0, num=config['evaluation']['num_thresholds'])

        # A list with all average prec. and recall by IoU
        precision_by_iou = []
        recall_by_iou = []
        f1_score_by_iou = []
        for threshold in thresholds:
            print(f"Evaluating threshold: {threshold} ...")
            test_data_generator.set_threshold(threshold)
            # A list with precision and recall per sample of testdata
            precision_by_testdata = []
            recall_by_testdata = []
            f1_score_by_testdata = []
            for sample in test_data_generator:
                try:
                    if config['explore_data']:
                        prediction, target, _ = sample
                    else:
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
            precision_by_iou.append(avg_prec)
            recall_by_iou.append(avg_rec)
            f1_score_by_iou.append(avg_f1)
            step_time = datetime.now() - overall_start_time
            print("Time elapsed: {}".format(step_time))

        data = [[x, y] for (x, y) in zip(recall_by_iou, precision_by_iou)]
        table = wandb.Table(data=data, columns=["Recall", "Precision"])
        wandb_logger.log({"PR curve": wandb.plot.line(table, "Recall", "Precision",
                                                      title=f"Precision-Recall over {len(test_data_generator)} samples")})
        wandb_logger.log({"F1": sum(f1_score_by_iou) / len(f1_score_by_iou)})


if __name__ == '__main__':
    main()
