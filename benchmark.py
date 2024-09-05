from stixel import StixelWorld, CameraInfo
import yaml
import glob
import os

with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

with open('sample/results/waymo_calib.yaml') as yaml_file:
    calib = yaml.load(yaml_file, Loader=yaml.FullLoader)
cam_info = CameraInfo(cam_mtx_k=calib['K'])

result_map = sorted(glob.glob(os.path.join(config['results_path'], '*.csv')))
for result in result_map:
    stixel_world: StixelWorld = StixelWorld.read(filepath=result,
                                                 camera_info=cam_info)
    stixel_world.save(filepath="sample/results",
                      binary=True,
                      incl_image=True)