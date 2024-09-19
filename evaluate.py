""" Evaluate, model + weights
"""
import glob
import importlib
import os.path
from collections import OrderedDict
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stixel as stx
import torch
import yaml
from PIL import Image
from einops import rearrange

import wandb
from dataloader import WaymoDataLoader
from metric import evaluate_sample_3dbbox
from utils.visualization import draw_stixel_and_bboxes

overall_start_time = datetime.now()

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)


def main():
    loader = WaymoDataLoader(data_dir=config['metric_data_path'],
                             first_only=True)
    precision = []
    recall = []

    run = wandb.init(project="StixelNExT-Pro",
                     job_type="analysis",
                     tags=["analysis"]
                     )
    artifact = run.use_artifact(f"{config['artifact']}", type='model')
    analysis_artifact = wandb.Artifact('model_analysis', type='evaluation', description="Analyzed model artifact")

    stxl_model = StixelModel(wandb_artifact=artifact)

    result_dir = os.path.join('results', stxl_model.checkpoint_name)
    os.makedirs(result_dir, exist_ok=True)

    for probability in np.arange(0.0, 1.0, 0.05):
        probab_result = {}
        probab_result['Stixel-Score'] = np.array([])
        probab_result['BBox-Score'] = np.array([])
        sample_results = {}
        index = 1
        for record in loader:
            sample_idx = 0
            print(f"Starting record with {len(record)} samples.")
            for sample in record:
                # if sample_idx == 12:
                start_time = datetime.now()
                # Inference a Stixel World
                start_inf = datetime.now()
                stxl_wrld = stxl_model.inference(sample.image, probability=probability, calib=sample.calib)
                print(f"Inference: {datetime.now() - start_inf}")
                # Apply the evaluation
                start_eval = datetime.now()
                results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(stxl_wrld, sample.bboxes)
                print(f"Evaluation: {datetime.now() - start_eval}")
                probab_result['Stixel-Score'] = np.append(probab_result['Stixel-Score'], results['Stixel-Score'])
                probab_result['BBox-Score'] = np.append(probab_result['BBox-Score'], results['BBox-Score'])
                sample_results[sample.name] = results
                sample_time = datetime.now() - start_time
                results_short = results.copy()
                results_short.pop('bbox_dist', None)
                print(
                    f"{sample.name} (idx={sample_idx}) with Stixel-Score:{results['Stixel-Score']} and BBox-Score: {results['BBox-Score']} within {sample_time}. {results_short}")
                if probability > 1.8:
                    draw_stixels_result(waymo_data=sample, stxl_wrld=stxl_wrld, stixel_pts=stixel_pts,
                                        stixel_colors=stixel_colors)
                sample_idx += 1
            step_time = datetime.now() - overall_start_time
            print("#####################################################################")
            print(
                f"Record-file {index}/ {len(loader)} evaluated with [Stixel-Score: {np.mean(probab_result['Stixel-Score'])} %/ BBox-Score of {np.mean(probab_result['BBox-Score'])} %]. Time elapsed: {step_time}")
            index += 1
        probab_score = np.mean(probab_result['Stixel-Score'])
        probab_bbox_score = np.mean(probab_result['BBox-Score'])

        df = pd.DataFrame.from_dict(sample_results, orient='index')
        df.index.name = 'Sample_ID'
        df.to_csv(os.path.join(result_dir,
                               f"{loader.name}-{config['results_name']}_PROB-{probability}_StixelScore-{probab_score}_bboxScore-{probab_bbox_score}.csv"))

        # sample: WaymoData = loader[0][0]
        # results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(sample.stixel_wrld, sample.laser_labels)
        # print(results)
        # Visualise point cloud
        # draw_stixel_and_bboxes(stixel_pts, stixel_colors, sample.laser_labels)
        print(
            f"Finished probability: {probability} with a Stixel-Score of {probab_score} % and a BBox-Score of {probab_bbox_score} % over {len(sample_results)} samples.")
        # probab.append(probability)
        # Precision/ recall
        precision.append(probab_score)
        recall.append(probab_bbox_score)
        # wandb log
        run.log({"precision": probab_score, "recall": probab_bbox_score})

    plt.figure()
    plt.plot(recall, precision, label='NN', marker='o', color='turquoise')
    plt.plot(1.155, 0.974, label='GT', marker='x', color='fuchsia')
    plt.xlabel('Probability')
    plt.ylabel('Score')
    name = f"{loader.name}-{config['results_name']} {stxl_model.checkpoint_name}"
    plt.title(name)
    plt.legend()
    plt.savefig(os.path.join(result_dir, name + '.png'))

    analysis_artifact.add_file(os.path.join(result_dir, name + '.png'))
    run.log_artifact(analysis_artifact)
    run.finish()


def draw_stixels_result(waymo_data, stxl_wrld, stixel_pts, stixel_colors):
    """
    Process an image sample, convert it to BGR, encode it, store it in stxl_wrld,
    draw stixels, and display the image.

    Args:
        waymo_data: An object containing the image and bounding boxes.
        stxl_wrld: An object that contains world information including the image.
    """
    img = np.array(waymo_data.image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, img_encoded = cv2.imencode('.png', img)
    stxl_wrld.image = img_encoded.tobytes()
    img_with_stixels = stx.draw_stixels_on_image(stxl_wrld)
    img_with_stixels.show()
    if len(stxl_wrld.stixel) != 0:
        draw_stixel_and_bboxes(stixel_pts, stixel_colors, waymo_data.bboxes)


def download_artifact_files(wandb_artifact: wandb.Artifact):
    path = os.path.join(os.getcwd(), "models", wandb_artifact.digest)
    os.makedirs(path, exist_ok=True)
    wandb_artifact.download(path)
    py_files = glob.glob(f'{path}/*.py')
    pth_files = glob.glob(f'{path}/*.pth')
    return py_files[0], pth_files[0]


class StixelModel:
    def __init__(self, wandb_artifact: wandb.Artifact = None):
        self.model_cfg = wandb_artifact.metadata
        model_filename, chckpt_filename = download_artifact_files(wandb_artifact)
        self.checkpoint_name = os.path.basename(os.path.splitext(chckpt_filename)[0])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        relative_import_path = os.path.relpath(model_filename, os.getcwd())
        module_name = relative_import_path.replace('/', '.').replace('.py', '')
        module = importlib.import_module(module_name)
        self.model, model_cfg = module.get_model(config=self.model_cfg)
        if self.model_cfg['mode'] == "segmentation":
            from models import revert_segm as revert_fn
        elif self.model_cfg['mode'] == "classification":
            from models import revert_class as revert_fn
        self.revert_fn = revert_fn

        self.model = self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(chckpt_filename)
        new_state_dict = OrderedDict()
        model_state = checkpoint['model_state_dict']
        for k, v in model_state.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        print("Loaded checkpoint '{}'".format(self.checkpoint_name))
        anchors_path = os.path.join('models', "depth_anchors.csv")
        if os.path.isfile(anchors_path):
            self.depth_anchors = pd.read_csv(anchors_path, index_col=0)
        else:
            self.depth_anchors = self._create_depth_bins(5, 50, 64)

    def inference(self, input_img: Image, probability, calib: stx.stixel_world_pb2.CameraInfo) -> stx.StixelWorld:
        input_tensor: torch.Tensor = torch.from_numpy(np.array(input_img)).to(torch.float32)
        input_tensor = rearrange(input_tensor, "h w c -> c h w")
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        output = output.cpu().detach()
        stxl_wrld: stx.StixelWorld = self.revert_fn(output[0], anchors=self.depth_anchors, prob=probability,
                                                    calib=calib)
        return stxl_wrld

    @staticmethod
    def _create_depth_bins(start=5, end=55, num_bins=192):
        bin_vals = np.linspace(start, end, num_bins)
        bin_mtx = np.tile(bin_vals, (240, 1))
        df = pd.DataFrame(bin_mtx)
        df = df.T
        df.columns = [str(i) for i in range(240)]
        return df


if __name__ == "__main__":
    main()
