import stixel as stx
import yaml
import glob
import os

with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


result_map = sorted(glob.glob(os.path.join("sample/data", '*.csv')))
for result in result_map:
    stixel_world: stx.StixelWorld = stx.read_csv(result,
                                                 camera_calib_file="sample/waymo_calib.yaml")
    stx.save(stixel_world, "sample/stxl/")
