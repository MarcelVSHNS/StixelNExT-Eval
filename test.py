import wandb
import yaml
import time
import numpy as np
from metrics.PrecisionRecall import evaluate_stixels, plot_precision_recall_curve, draw_stixel_on_image_prcurve

# TODO: just apply metrics and scores for wandb
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def main():
    # Create an export of analysed data incl. samples and ROC curve
    if config['logging']['activate']:
        # Init the logger
        # e.g. StixelNExT_ancient-silence-25_epoch-94_loss-0.09816327691078186.pth
        epochs = config['weights_file'].split('_')[2].split('-')[1]
        checkpoint = config['weights_file'].split('_')[1]
        wandb_logger = wandb.init(project=config['logging']['project'],
                                  config={
                                      "architecture": type(model).__name__,
                                      "dataset": testing_data.name,
                                      "checkpoint": checkpoint,
                                      "epochs": epochs
                                  },
                                  tags=["metrics", "testing"]
                                  )

        stixel_reader = StixelNExTInterpreter(detection_threshold=config['pred_threshold'],
                                              hysteresis_threshold=config['pred_threshold'] - 0.05)
        thresholds = np.linspace(0.1, 1.0, num=10)
        prec_ious = []
        recall_ious = []
        for iou in thresholds:
            prec_batches = []
            recall_batches = []
            for batch_idx, (samples, targets, images) in enumerate(testing_dataloader):
                # TODO: use loaded data
                samples = samples.to(device)
                start = time.process_time_ns()
                output = model(samples)
                t_infer = time.process_time_ns() - start
                # fetch data from GPU
                output = output.cpu().detach()
                prec_samples = []
                recall_samples = []
                for i in range(output.shape[0]):
                    target_stixel = stixel_reader.extract_stixel_from_prediction(targets[i])
                    prediction_stixel = stixel_reader.extract_stixel_from_prediction(output[i])
                    precision, recall, best_matches = evaluate_stixels(prediction_stixel, target_stixel, iou_threshold=iou)
                    prec_samples.append(precision)
                    recall_samples.append(recall)
                prec_batches.append(sum(prec_samples)/ len(prec_samples))
                recall_batches.append(sum(recall_samples)/ len(recall_samples))

            prec_ious.append(sum(prec_batches)/ len(prec_batches))
            recall_ious.append(sum(recall_batches) / len(recall_batches))
        f1_scores = [2 * p * r / (p + r) if (p + r) != 0 else 0 for p, r in zip(prec_ious, recall_ious)]
        data = [[x, y] for (x, y) in zip(recall_ious, prec_ious)]
        table = wandb.Table(data=data, columns=["Recall", "Precision"])
        wandb_logger.log({"PR curve": wandb.plot.line(table, "Recall", "Precision",
                                                       title=f"Precision-Recall over {testing_data.__len__()} samples")})
        wandb_logger.log({"F1": sum(f1_scores)/ len(f1_scores)})
if __name__ == '__main__':
    main()
