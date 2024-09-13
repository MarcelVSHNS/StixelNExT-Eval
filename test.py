from dataloader import WaymoDataLoader, WaymoData
from metric import evaluate_sample_3dbbox
from utils import draw_stixel_and_bboxes
import stixel as stx
import random
import yaml


if __name__ == "__main__":
    with open('config.yaml') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    loader = WaymoDataLoader(data_dir=config['metric_data_path'],
                             result_dir=config['results_path'],
                             first_only=True)
    idx = random.randint(0, len(loader) - 1)
    print(f"Random index: {idx}")
    drive = loader[idx]
    sample: WaymoData = drive[0]

    results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(sample.stxl_wrld, sample.bboxes)
    stxl_img = stx.draw_stixels_on_image(stxl_wrld=sample.stxl_wrld)
    stxl_img.show()
    print(results)
    # Visualise point cloud
    draw_stixel_and_bboxes(stixel_pts, stixel_colors, sample.bboxes)
