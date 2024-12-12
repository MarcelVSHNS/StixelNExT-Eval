# import multiprocessing
import os
import os.path
from datetime import datetime
import random

import stixel as stx
import yaml
import torch

from dataloader import WaymoDataLoader, StixelModel
from metric import evaluate_sample_3dbbox, evaluate_sample_segmentation
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
    if config["device"] == "gpu":
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device('cpu')
    stxl_model = StixelModel(device=dev, n_cand=config["n_cand"])
    stxl_model.info()

    # local results directory
    result_dir = os.path.join('sample_results', stxl_model.checkpoint_name)
    os.makedirs(result_dir, exist_ok=True)
    idx = random.randint(0, len(loader) - 1)
    print(f"random Idx: {idx}")
    sample = loader[143][0] #31 for default sample
    probability = config["explore_thres"]

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
    prec = results["Stixel-Score"]
    recall = results["BBox-Score"]
    print(f"F1: {calculate_f1(prec, recall)} \t Precision: {prec} \t Recall: {recall}")
    if sample.panoptics:
        segmentation_score, segmentation_img = evaluate_sample_segmentation(stxl_wrld, sample.semantic_label, image=sample.image.convert('L'))
        print(f"Segmentation mIoU: {segmentation_score}")
        segmentation_img.show()

    stxl_wrld = stx.add_image(stxl_wrld, sample.image)
    sample.image.show()
    print(results)
    # img = stx.draw_stixels_on_image(stxl_wrld)
    # img.show()
    if stixel_pts:
        stxl_wrld_clustered = stx.attach_dbscan_clustering(stxl_wrld, min_samples=1)
        stxl_img = stx.draw_stixels_on_image(stxl_wrld_clustered, instances=True)
        stxl_img.show()
        draw_stixel_and_bboxes(stixel_pts, stixel_colors, sample.bboxes)


def calculate_f1(precision: float, recall: float):
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


if __name__ == "__main__":
    main()
