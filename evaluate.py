"""
TODO: WANDB Integration
"""
import os.path
from dataloader import WaymoDataLoader, WaymoData
from metric import evaluate_sample_3dbbox
from utils import draw_stixel_and_bboxes
from stixel.utils import draw_stixels_on_image
import numpy as np
import pandas as pd
import yaml
from datetime import datetime

overall_start_time = datetime.now()


if __name__ == "__main__":
    with open('config.yaml') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    loader = WaymoDataLoader(data_dir=config['metric_data_path'],
                             result_dir=config['results_path'],
                             first_only=False)

    overall_result = {}
    overall_result['Stixel-Score'] = np.array([])
    overall_result['BBox-Score'] = np.array([])
    sample_results = {}
    index = 1
    for record in loader:
        sample_idx = 0
        print(f"Starting record with {len(record)} samples.")
        for sample in record:
            # if sample_idx == 12:
            start_time = datetime.now()
            results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(sample.stixel_wrld, sample.laser_labels)
            overall_result['Stixel-Score'] = np.append(overall_result['Stixel-Score'], results['Stixel-Score'])
            overall_result['BBox-Score'] = np.append(overall_result['BBox-Score'], results['BBox-Score'])
            sample_results[sample.name] = results
            sample_time = datetime.now() - start_time
            results_short = results.copy()
            results_short.pop('bbox_dist', None)
            print(f"{sample.name} (idx={sample_idx}) with Stixel-Score:{results['Stixel-Score']} and BBox-Score: {results['BBox-Score']} within {sample_time}. {results_short}")
            #if sample_idx == 9:
            #    img = draw_stixels_on_image(sample.image, sample.stixel_wrld.stixel)
            #    img.show()
            #    draw_stixel_and_bboxes(stixel_pts, stixel_colors, sample.laser_labels)
            sample_idx += 1
        step_time = datetime.now() - overall_start_time
        print("#####################################################################")
        print(f"Record-file {index}/ {len(loader)} evaluated with [Stixel-Score: {np.mean(overall_result['Stixel-Score'])} %/ BBox-Score of {np.mean(overall_result['BBox-Score'])} %]. Time elapsed: {step_time}")
        index += 1
    overall_score = np.mean(overall_result['Stixel-Score'])
    overall_bbox_score = np.mean(overall_result['BBox-Score'])

    df = pd.DataFrame.from_dict(sample_results, orient='index')
    df.index.name = 'Sample_ID'
    df.to_csv(os.path.join(config['results_path'], f"{loader.name}-{config['results_name']}_StixelScore-{overall_score}_bboxScore-{overall_bbox_score}.csv"))

    # sample: WaymoData = loader[0][0]
    # results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(sample.stixel_wrld, sample.laser_labels)
    # print(results)
    # Visualise point cloud
    # draw_stixel_and_bboxes(stixel_pts, stixel_colors, sample.laser_labels)
    print(f"Finished with a Stixel-Score of {overall_score} % and a BBox-Score of {overall_bbox_score} % over {len(sample_results)} samples.")
