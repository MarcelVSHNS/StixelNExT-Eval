from metric.bbox_iou import evaluate_sample_3dbbox
from utils.visualization import draw_mmdet_bboxes_on_image
import yaml
from dataloader import WaymoDataLoader, PGDPredictor
import numpy as np


with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)


def main():
    loader = WaymoDataLoader(data_dir=config["metric_data_path"],
                                 first_only=True)
    sample = loader[143][0]

    config_path = "models/pgd_r101_fpn_gn-head_dcn_8xb3-2x_waymoD3-fov-mono3d.py"
    chckpt_path = "models/chckpts/epoch_20.pth"
    pgd_model = PGDPredictor(cfg_path=config_path, chckpt_path=chckpt_path, thres=0.3)
    # sample.image, sample.calib.K.reshape(3, 3)
    print(sample.k2)
    results = pgd_model.predict(sample.image, sample.k2)
    # corners_3d = results.corners  # (N, 8, 3)
    #np.save('pred_corners_3d.npy', corners_3d)
    T = np.array(sample.calib.T).reshape(4, 4)
    results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(sample.bboxes, pred_bboxes=results, trans_mtx=T, bbox_mode=True)
    print(results)



if __name__ == "__main__":
    main()