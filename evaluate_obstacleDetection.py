import wandb
import yaml
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
import os
from metrics.ObstacleDetection import ObstacleMetric, visualize_stixels_on_image
from resultloader.MultiStixelStereo import StereoStixelLoader as Dataloader


def main():
    test_data_generator = Dataloader(obstacle_detection_mode=True)
    metric = ObstacleMetric()
    pred, targ, image = test_data_generator[1]
    print(metric.evaluate(pred, targ))
    visualize_stixels_on_image(image, pred, targ, name=f"Stereo_")
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
                                  tags=["Stereo", "obstacle detection"]
                                  )
    i = 0
    for pred, targ, image in test_data_generator:
        i += 1
        score = metric.evaluate(pred, targ)
        # test_data_generator.stixel_reader.show_stixel(image, stixel_list=pred, color=[255, 0, 0])
        if i % 10 == 0:
            print(f"Current score: {score}")
        if i in [100, 200, 300, 400]:
            visualize_stixels_on_image(image, pred, targ, name=f"Stereo_{i}")
        #wandb_logger.log({"Obstacle Score": metric.get_score()})


if __name__ == '__main__':
    main()
