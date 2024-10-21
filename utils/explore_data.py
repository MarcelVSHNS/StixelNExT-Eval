# import multiprocessing
import os
import os.path
from datetime import datetime

import stixel as stx
import yaml

from dataloader import WaymoDataLoader, StixelModel
from metric import evaluate_sample_3dbbox
from utils.visualization import draw_stixel_and_bboxes

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

overall_start_time = datetime.now()
os.environ["WANDB_REPORT_API_ENABLE_V2"] = "True"
os.environ["WANDB_REPORT_API_DISABLE_MESSAGE"] = "True"


def main():
    loader = WaymoDataLoader(data_dir=config["metric_data_path"],
                             first_only=True)
    # model
    stxl_model = StixelModel()
    stxl_model.info()

    # local results directory
    result_dir = os.path.join('sample_results', stxl_model.checkpoint_name)
    os.makedirs(result_dir, exist_ok=True)
    sample = loader[31][0]
    probability = 0.55

    # Inference a Stixel World
    start_time = datetime.now()
    stxl_infer = stxl_model.inference(sample.image)
    print(f"inference time: {datetime.now() - start_time}")
    start_time = datetime.now()
    stxl_wrld = stxl_model.revert(stxl_infer, probability=probability, calib=sample.calib)
    print(f"reverting time: {datetime.now() - start_time}")
    # if len(stxl_wrld.stixel) > 2000:
    #     continue
    # print(f"Inference: {datetime.now() - start_inf}")
    # Apply the evaluation
    results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(stxl_wrld, sample.bboxes)
    stxl_wrld = stx.add_image(stxl_wrld, sample.image)
    print(results)
    img = stx.draw_stixels_on_image(stxl_wrld)
    img.show()
    draw_stixel_and_bboxes(stixel_pts, stixel_colors, sample.bboxes)


if __name__ == "__main__":
    main()
