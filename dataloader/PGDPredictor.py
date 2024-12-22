from mmengine import Config
from mmdet3d.apis import init_model, inference_mono_3d_detector
import json
import numpy as np
from collections import namedtuple
import math

class PGDPredictor:
    def __init__(self, cfg_path: str, chckpt_path: str, thres=0.5):
        self.cfg = Config.fromfile(cfg_path)
        self.model = init_model(self.cfg, chckpt_path, device='cuda:0')
        self.thres = thres

    def predict(self, img, calib):
        #K = np.hstack((calib, np.array([[0], [0], [0]])))
        K = calib
        img.save('sample.jpg')
        img_path = 'sample.jpg'
        ann_file = self._get_ann_file(img_path, K)
        results = inference_mono_3d_detector(
            self.model,
            imgs=[img_path],
            ann_file=ann_file,
            cam_type='CAM_FRONT'
        )
        # print(results[0])
        return (self._filter_boxes(results[0]).cpu().detach()).corners

    def _filter_boxes(self, result):
        mask = result.pred_instances_3d.scores_3d > self.thres
        filtered_boxes = result.pred_instances_3d.bboxes_3d[mask]
        return filtered_boxes

    def _get_ann_file(self, img_path: str, calib: np.ndarray):
        data_list = {
            "data_list": [
                {
                    "images": {
                        "CAM_FRONT": {
                            "img_path": img_path,
                            "cam2img": calib.tolist()
                        }
                    }
                }
            ]
        }
        output_file = "ann_file.json"
        with open(output_file, "w") as f:
            json.dump(data_list, fp=f, indent=4)
        return output_file
