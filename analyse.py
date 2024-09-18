import multiprocessing
import os
import os.path
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import yaml

import wandb
from dataloader import WaymoDataLoader, StixelModel
from metric import evaluate_sample_3dbbox
from utils import draw_stixel_and_bboxes

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

overall_start_time = datetime.now()


def main():
    loader = WaymoDataLoader(data_dir=config["metric_data_path"],
                             first_only=True)
    # logger
    logger = wandb.init(project="StixelNExT-Pro",
                        job_type="analysis",
                        tags=["analysis"]
                        )
    # model
    stxl_model = StixelModel()
    # local results directory
    result_dir = os.path.join('sample_results', stxl_model.checkpoint_name)
    os.makedirs(result_dir, exist_ok=True)

    index_list = list(range(0, len(loader), 8))
    analyse_partial = partial(analyse,
                              stxl_model=stxl_model,
                              dataloader=loader
                              )
    with multiprocessing.Pool() as pool:
        results = pool.map(analyse_partial, index_list)

    # save every precision/ recall of every sample and probability
    # e.g. a dict with probability as key and a list of precisions (by sample) as value
    precision = {}
    recall = {}
    for sample in results:
        for probab, result in sample['results'].items():
            precision[probab].append(result['precision'])
            recall[probab].append(result['recall'])

    # draw precision recall curve
    plt.figure()
    plt.plot(recall, precision, label='NN', marker='o', color='turquoise')
    plt.plot(1.155, 0.974, label='GT', marker='x', color='fuchsia')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    name = f"{loader.name}-{config['results_name']} {stxl_model.checkpoint_name}"
    plt.title(name)
    plt.legend()
    plt.savefig(os.path.join(result_dir, name + '.png'))

    # draw best in 3d
    best_prob = None
    max_sum = -1
    for prob, metrics in probab_result.items():
        precision = metrics['precision']
        recall = metrics['recall']

        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0

        if f1_score > max_f1:
            max_f1 = f1_score
            best_prob = prob
    print(
        f"Probability: {best_prob} with prec: {probab_result[best_prob]['precision']} with recall: {probab_result[best_prob]['recall']}")
    draw_stixel_and_bboxes(probab_result[best_prob]['pts'], probab_result[best_prob]['colors'], sample.bboxes)

    logger.finish()


def analyse(sample_idx: float, stxl_model: StixelModel, dataloader: WaymoDataLoader):
    sample = dataloader[sample_idx][0]
    result_list = {}
    for probability in np.arange(0.0, 1.0, 0.05):
        start_time = datetime.now()
        # Inference a Stixel World
        start_inf = datetime.now()
        stxl_wrld = stxl_model.inference(sample.image, probability=probability, calib=sample.calib)
        if len(stxl_wrld.stixel) > 2000:
            continue
        print(f"Inference: {datetime.now() - start_inf}")
        # Apply the evaluation
        start_eval = datetime.now()
        results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(stxl_wrld, sample.bboxes)
        print(f"Evaluation: {datetime.now() - start_eval}")
        result_list[probability] = {"precision": results['Stixel-Score'],
                                    "recall": results['BBox-Score'],
                                    "f1_score": calculate_f1(precision=results['Stixel-Score'],
                                                             recall=results['BBox-Score']),
                                    "results": results,
                                    "pts": stixel_pts,
                                    "colors": stixel_colors,
                                    "stxl_wrld": stxl_wrld}
        sample_time = datetime.now() - start_time
        results_short = results.copy()
        results_short.pop('bbox_dist', None)
        print(
            f"{sample.name} @ {probability} with Precision:{results['Stixel-Score']} and Recall: {results['BBox-Score']} within {sample_time}. {results_short}")
        print("#####################################################################")
        # logs
        # wandb.log({f"Sample @ {probability}": wandb.Image(stxl_img)})
        # logger.log({"precision": results['Stixel-Score'], "recall": results['BBox-Score']})
    highest_f1_score = -1
    best_probability = None
    for probability, values in result_list.items():
        if values["f1_score"] > highest_f1_score:
            highest_f1_score = values["f1_score"]
            best_probability = probability
    return {"idx": sample_idx,
            "image": sample.image,
            "pick": best_probability,
            "results": result_list}


def calculate_f1(precision: float, recall: float):
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


if __name__ == "__main__":
    main()
