import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

from resultloader.EvaluationDataloader import EvaluationDataloader, read_stixel_from_csv
from dataloader.stixel_multicut import MultiCutStixelData
from dataloader.stixel_multicut_interpreter import StixelNExTInterpreter
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
        self.dataset_dir = os.path.join(config['data_path'], config['dataset'])
        super().__init__(os.path.join(self.dataset_dir, "testing", "targets_from_lidar"))
        self.prediction_list = []
        self._create_predictions()
        # Stixel Interpreter
        self.stixel_reader = StixelNExTInterpreter()
        # Check-ups
        if len(self.prediction_list) != len(self.target_list):
            print(f"INFO: Inconsistent number of predictions[{len(self.prediction_list)}] and targets[{len(self.target_list)}]")

    def __getitem__(self, idx):
        pred_stx = self.stixel_reader.extract_stixel_from_prediction(self.prediction_list[idx]['prediction'])
        targ_stx = read_stixel_from_csv(os.path.join(self.target_folder, self.prediction_list[idx]['filename'] + ".csv"))
        return pred_stx, targ_stx

    def __len__(self):
        return len(self.prediction_list)

    def set_threshold(self, threshold, hysteresis=0.05):
        self.stixel_reader.s1 = threshold
        self.stixel_reader.s2 = threshold - hysteresis

    def _create_predictions(self):
        # Test dataloader
        testing_data = MultiCutStixelData(data_dir=self.dataset_dir,
                                          phase='testing',
                                          transform=feature_transform,
                                          target_transform=None,
                                          return_name=True)
        # Model
        model = ConvNeXt(stem_features=config['nn']['stem_features'],
                         depths=config['nn']['depths'],
                         widths=config['nn']['widths'],
                         drop_p=config['nn']['drop_p'],
                         target_height=int(testing_data.img_size['height'] / config['grid_step']),
                         target_width=int(testing_data.img_size['width'] / config['grid_step']),
                         out_channels=2).to(device)
        # Load weights
        weights_file = config['weights_file']
        checkpoint = os.path.splitext(weights_file)[0]  # checkpoint without ending
        run = checkpoint.split('_')[1]
        model.load_state_dict(
            torch.load(f=os.path.join(self.dataset_dir, "testing", "weights_from_StixelNExT", weights_file),
                       map_location=torch.device(device)))

        for idx in range(len(testing_data)):
            sample, target, name = testing_data[idx]
            sample = sample.unsqueeze(0)
            sample = sample.to(device)
            output = model(sample)
            # fetch data from GPU
            output = output.cpu().detach()
            self.prediction_list.append({"filename": name, "prediction": output})
        print(f"Predictions with Checkpoint {weights_file} created!")
