import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

from resultloader import EvaluationDataloader, read_stixel_from_csv
from dataloader.stixel_multicut import MultiCutStixelData
from dataloader.stixel_multicut_interpreter import StixelNExTInterpreter
from torch.utils.data import DataLoader
from models.ConvNeXt import ConvNeXt
import torch
import os
if config['dataset'] == "kitti":
    from dataloader.stixel_multicut import feature_transform_resize as feature_transform
else:
    feature_transform = None

device = "cuda" if torch.cuda.is_available() else "cpu"


class StixelNExTLoader(EvaluationDataloader):
    def __init__(self):
        # Targets
        dataset_dir = os.path.join(config['data_path'], config['dataset'])
        super().__init__(os.path.join(dataset_dir, "testing", "targets_from_lidar"))
        # Test dataloader
        self.testing_data = MultiCutStixelData(data_dir=dataset_dir,
                                               phase='testing',
                                               transform=feature_transform,
                                               target_transform=None,
                                               return_name=True)
        # Model
        self.model = ConvNeXt(stem_features=config['nn']['stem_features'],
                              depths=config['nn']['depths'],
                              widths=config['nn']['widths'],
                              drop_p=config['nn']['drop_p'],
                              target_height=int(self.testing_data.img_size['height'] / config['grid_step']),
                              target_width=int(self.testing_data.img_size['width'] / config['grid_step']),
                              out_channels=2).to(device)
        # Load weights
        self.weights_file = config['weights_file']
        self.checkpoint = os.path.splitext(self.weights_file)[0]  # checkpoint without ending
        self.run = self.checkpoint.split('_')[1]
        self.model.load_state_dict(torch.load(os.path.join(dataset_dir, "testing", "weights_from_StixelNExT", self.weights_file)))
        # Stixel Interpreter
        self.stixel_reader = StixelNExTInterpreter()
        # Check-ups
        if len(self.testing_data) != len(self.target_list):
            print(f"INFO: Inconsistent number of predictions[{len(self.testing_data)}] and targets[{len(self.target_list)}]")

    def __getitem__(self, idx):
        sample, target, filename = self.testing_data[idx]
        sample = sample.to(device)
        output = self.model(sample)
        # fetch data from GPU
        pred_stx = output.cpu().detach()
        targ_stx = read_stixel_from_csv(os.path.join(self.target_folder, filename))
        return pred_stx, targ_stx

    def __len__(self):
        return len(self.testing_data)
