import os
import os.path
from collections import OrderedDict

import numpy as np
import pandas as pd
import stixel as stx
import torch
from PIL import Image
from einops import rearrange
from torchinfo import summary


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

    @staticmethod
    def _create_depth_bins(start=5, end=55, num_bins=192):
        bin_vals = np.linspace(start, end, num_bins)
        bin_mtx = np.tile(bin_vals, (240, 1))
        df = pd.DataFrame(bin_mtx)
        df = df.T
        df.columns = [str(i) for i in range(240)]
        return df
