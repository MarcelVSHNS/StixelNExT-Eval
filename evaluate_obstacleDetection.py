import wandb
import yaml
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
import os
import numpy as np
from metrics.ObstacleDetection import ObstacleMetric, visualize_stixels_on_image
if config['evaluation']['model'] == "StixelNExT":
    from resultloader import StixelNExTLoader as Dataloader


def main():
    test_data_generator = Dataloader(obstacle_detection_mode=True, exploring=True)
    metric = ObstacleMetric()
    if config['logging']['activate']:
        # Init the logger
        # e.g. StixelNExT_ancient-silence-25_epoch-94_loss-0.09816327691078186.pth predictions_from_stereo
        run = config['weights_file']
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
                                  tags=["StixelNExT", "obstacle detection"]
                                  )
        i = 0
        scores = []
        for pred, targ, image in test_data_generator:
            i += 1
            score = metric.evaluate(pred, targ)
            # test_data_generator.stixel_reader.show_stixel(image, stixel_list=pred, color=[255, 0, 0])
            wandb_logger.log({"score": score})
            scores.append(score)
            if i % 10 == 0:
                print(f"Current score: {score}")
            if i in [100, 200, 300, 400]:
                visualize_stixels_on_image(image, pred, targ, name=f"StixelNExT_{i}", stixel_width=config['grid_step'])
        wandb_logger.log({"Std Sigma": np.std(scores)})
        wandb_logger.log({"Obstacle Score": metric.get_score()})


if __name__ == '__main__':
    main()
