import glob
import importlib
import os
import os.path
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd
import stixel as stx
import torch
from PIL import Image
from einops import rearrange
from torchinfo import summary
from wandb.apis.importers import wandb

from models.base.ConvNeXt_pretrained import get_model


def download_artifact_files(wandb_artifact: wandb.Artifact):
    path = os.path.join(os.getcwd(), "models", wandb_artifact.digest)
    os.makedirs(path, exist_ok=True)
    wandb_artifact.download(path)
    py_files = glob.glob(f'{path}/*.py')
    pth_files = glob.glob(f'{path}/*.pth')
    return py_files[0], pth_files[0]


class StixelModel:
    def __init__(self, artifact: Optional[wandb.Artifact] = None, device: torch.device = torch.device('cpu')):
        # load configuration and model
        if artifact is None:
            self._load_from_config()
        else:
            self._load_from_artifact(artifact)
        self.device = device
        self.model = self.model.to(self.device)
        # Load Checkpoint
        checkpoint = torch.load(self.chckpt_filename, weights_only=True)
        new_state_dict = OrderedDict()
        model_state = checkpoint['model_state_dict']
        for k, v in model_state.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        # print("Loaded checkpoint '{}'".format(self.checkpoint_name))
        self.model.eval()
        # Depth Anchors and revert function
        self.depth_anchors = self._create_depth_bins(5, 69, 64)
        if self.model_cfg['mode'] == "segmentation":
            from models import revert_segm as revert_fn
        elif self.model_cfg['mode'] == "classification":
            from models import revert_class as revert_fn
        self.revert_fn = revert_fn

    def inference(self, input_img: Image) -> stx.StixelWorld:
        input_tensor: torch.Tensor = torch.from_numpy(np.array(input_img)).to(torch.float32)
        input_tensor = rearrange(input_tensor, "h w c -> c h w")
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        self.model = self.model.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        output = output.cpu().detach()
        return output[0]

    def revert(self, stixel_infer, probability, calib: stx.stixel_world_pb2.CameraInfo):
        stxl_wrld: stx.StixelWorld = self.revert_fn(stixel_infer,
                                                    anchors=self.depth_anchors,
                                                    prob=probability,
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

    def _load_from_config(self):
        self.model_cfg = {"C": 96,
                          "B": [3, 3, 9, 3],
                          "stem_features": 64,
                          "n_candidates": 64,
                          "i_attributes": 3,
                          "n_bins": 64,
                          "mode": "segmentation"}
        self.checkpoint_name = "StixelNExT-Pro_iconic-aardvark-221_25"
        self.chckpt_filename = "models/chckpts/StixelNExT-Pro_iconic-aardvark-221_25.pth"
        self.checkpoint_name = os.path.basename(os.path.splitext(self.chckpt_filename)[0])
        self.model, _ = get_model(config=self.model_cfg)

    def _load_from_artifact(self, artifact: wandb.Artifact):
        self.model_cfg = artifact.metadata
        model_filename, self.chckpt_filename = download_artifact_files(artifact)
        relative_import_path = os.path.relpath(model_filename, os.getcwd())
        module_name = relative_import_path.replace('/', '.').replace('.py', '')
        module = importlib.import_module(module_name)
        self.checkpoint_name = os.path.basename(os.path.splitext(self.chckpt_filename)[0])
        self.model, _ = module.get_model(config=self.model_cfg)

    def info(self):
        summary(self.model, (1, 3, 1280, 1920), device=torch.device('cpu'))
        print(f"Models runs on {self.device}")
        print(f"Checkpoint: {self.checkpoint_name}")
