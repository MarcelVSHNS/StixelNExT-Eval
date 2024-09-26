import os
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
from torchinfo import summary

import wandb
from dataloader import WaymoDataLoader
from metric import evaluate_sample_3dbbox
from utils import draw_stixel_and_bboxes

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

overall_start_time = datetime.now()


def main():
    loader = WaymoDataLoader(data_dir="samples",
                             first_only=True)
    sample = loader[0][0]
    precision = []
    recall = []

    run = wandb.init(project="StixelNExT-Pro",
                     job_type="analysis",
                     tags=["analysis"]
                     )
    # artifact = run.use_artifact(f"{config['artifact']}", type='model')

    stxl_model = StixelModel()

    result_dir = os.path.join('sample_results', stxl_model.checkpoint_name)
    os.makedirs(result_dir, exist_ok=True)

    probab_result = {}
    for probability in np.arange(0.55, 0.95, 0.05):
        start_time = datetime.now()
        # Inference a Stixel World
        start_inf = datetime.now()
        stxl_wrld = stxl_model.inference(sample.image, probability=probability, calib=sample.calib)
        print(f"Inference: {datetime.now() - start_inf}")
        # Apply the evaluation
        start_eval = datetime.now()
        results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(stxl_wrld, sample.bboxes)
        print(f"Evaluation: {datetime.now() - start_eval}")
        probab_result[probability] = {"precision": results['Stixel-Score'],
                                      "recall": results['BBox-Score'],
                                      "results": results,
                                      "pts": stixel_pts,
                                      "colors": stixel_colors,
                                      "stxl_wrld": stxl_wrld}
        sample_time = datetime.now() - start_time
        results_short = results.copy()
        results_short.pop('bbox_dist', None)
        print(
            f"{sample.name} @ {probability} with Precision:{results['Stixel-Score']} and Recall: {results['BBox-Score']} within {sample_time}. {results_short}")
        print("#####################################################################")
        # Precision/ recall
        precision.append(results['Stixel-Score'])
        recall.append(results['BBox-Score'])
        # logs
        img = np.array(sample.image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        success, img_encoded = cv2.imencode('.png', img)
        stxl_wrld.image = img_encoded.tobytes()
        stxl_img = stx.draw_stixels_on_image(stxl_wrld)
        wandb.log({f"Sample @ {probability}": wandb.Image(stxl_img)})
        run.log({"precision": results['Stixel-Score'], "recall": results['BBox-Score']})

    # draw precision recall curve
    plt.figure()
    plt.plot(recall, precision, label='NN', marker='o', color='turquoise')
    plt.plot(1.155, 0.974, label='GT', marker='x', color='fuchsia')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    name = f"{loader.name}-{config['results_name']} {stxl_model.checkpoint_name}"
    plt.title(name)
    plt.legend()
    plt.savefig(os.path.join(result_dir, name + '.png'))

    # draw best in 3d
    best_prob = None
    max_sum = -1
    for prob, metrics in probab_result.items():
        precision = metrics['precision']
        recall = metrics['recall']

        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0

        if f1_score > max_f1:
            max_f1 = f1_score
            best_prob = prob
    print(
        f"Probability: {best_prob} with prec: {probab_result[best_prob]['precision']} with recall: {probab_result[best_prob]['recall']}")
    draw_stixel_and_bboxes(probab_result[best_prob]['pts'], probab_result[best_prob]['colors'], sample.bboxes)

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


class StixelModel:
    def __init__(self):
        self.model_cfg = {"C": 96,
                          "B": [3, 3, 9, 3],
                          "stem_features": 64,
                          "n_candidates": 64,
                          "i_attributes": 3,
                          "n_bins": 64,
                          "mode": "segmentation"}
        self.checkpoint_name = "StixelNExT-Pro_iconic-aardvark-221_25"
        chckpt_filename = "models/chckpts/StixelNExT-Pro_iconic-aardvark-221_25.pth"
        from models.base.ConvNeXt_pretrained import get_model
        self.checkpoint_name = os.path.basename(os.path.splitext(chckpt_filename)[0])
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.model, _ = get_model(config=self.model_cfg)
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
            self.depth_anchors = self._create_depth_bins(5, 69, 64)
        summary(self.model, (1, 3, 1280, 1920), device=torch.device('cpu'))

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

    def revert(self):
        pass

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
