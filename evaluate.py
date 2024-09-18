""" Evaluate, model + weights
"""
import multiprocessing
import os.path
from datetime import datetime
from functools import partial
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import wandb
from dataloader import WaymoDataLoader, StixelModel
from metric import evaluate_sample_3dbbox

overall_start_time = datetime.now()

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)


def main():
    loader = WaymoDataLoader(data_dir=config['metric_data_path'],
                             first_only=False)
    # create logger
    logger = wandb.init(project="StixelNExT-Pro",
                        job_type="analysis",
                        tags=["analysis"]
                        )
    artifact = logger.use_artifact(f"{config['artifact']}", type='model')
    # create model
    stxl_model = StixelModel(artifact=artifact)
    # create results_folder
    result_dir = os.path.join('results', stxl_model.checkpoint_name)
    os.makedirs(result_dir, exist_ok=True)

    # multiprocess every probability
    probability = np.arange(config["from"], config["to"] + config["in"], config["in"])
    evaluate_partial = partial(evaluate,
                               dataloader=loader,
                               stxl_model=stxl_model,
                               logger=logger,
                               result_dir=result_dir)
    with multiprocessing.Pool() as pool:
        results = pool.map(evaluate_partial, probability)

    # organize results in dict
    probabilities = []
    precisions = []
    recalls = []
    for result in results:
        probabilities.append(result['probability'])
        precisions.append(result['precision'])
        recalls.append(result['recall'])
    # create figure: Precision/ Recall
    plt.figure()
    plt.plot(recalls, precisions, label='NN', marker='o', color='turquoise')
    plt.plot(1.155, 0.974, label='GT', marker='x', color='fuchsia')
    plt.xlabel('Probability')
    plt.ylabel('Score')
    name = f"{loader.name}-{config['results_name']} {stxl_model.checkpoint_name}"
    plt.title(name)
    plt.legend()
    plt.savefig(os.path.join(result_dir, name + '.png'))
    f1_prec = np.mean(precisions)
    f1_recall = np.mean(recalls)
    f1_score = calculate_f1(precision=f1_prec, recall=f1_recall)
    logger.log({"Mean_F1-Score": f1_score})
    logger.finish()


def evaluate(probability: float,
             dataloader: WaymoDataLoader,
             stxl_model: StixelModel,
             logger: wandb.run,
             result_dir: str
             ) -> Dict[str, float]:
    probab_result = {'Stixel-Score': np.array([]), 'BBox-Score': np.array([])}
    sample_results = {}
    index = 1
    for record in dataloader:
        sample_idx = 0
        print(f"Starting record with {len(record)} samples.")
        for sample in record:
            start_time = datetime.now()
            # Inference a Stixel World
            stxl_wrld = stxl_model.inference(sample.image, probability=probability, calib=sample.calib)
            # Apply the evaluation
            start_eval = datetime.now()
            results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(stxl_wrld, sample.bboxes)
            print(f"Evaluation: {datetime.now() - start_eval}")
            probab_result['Stixel-Score'] = np.append(probab_result['Stixel-Score'], results['Stixel-Score'])
            probab_result['BBox-Score'] = np.append(probab_result['BBox-Score'], results['BBox-Score'])
            sample_results[sample.name] = results
            sample_time = datetime.now() - start_time
            results_short = results.copy()
            results_short.pop('bbox_dist', None)
            print(
                f"{sample.name} (idx={sample_idx}) with Stixel-Score:{results['Stixel-Score']} and BBox-Score: {results['BBox-Score']} within {sample_time}. {results_short}")
            sample_idx += 1
            logger.log({f"TotalPrecision@{probability}": np.mean(probab_result['Stixel-Score']),
                        f"TotalRecall@{probability}": np.mean(probab_result['BBox-Score'])})
        step_time = datetime.now() - overall_start_time
        print("#####################################################################")
        print(
            f"Record-file {index}/ {len(dataloader)} evaluated with [Stixel-Score: {np.mean(probab_result['Stixel-Score'])} %/ BBox-Score of {np.mean(probab_result['BBox-Score'])} %]. Time elapsed: {step_time}")
        index += 1
    probab_score = np.mean(probab_result['Stixel-Score'])
    probab_bbox_score = np.mean(probab_result['BBox-Score'])

    df = pd.DataFrame.from_dict(sample_results, orient='index')
    df.index.name = 'Sample_ID'
    df.to_csv(os.path.join(result_dir,
                           f"{dataloader.name}-{config['results_name']}_PROB-{probability}_StixelScore-{probab_score}_bboxScore-{probab_bbox_score}.csv"))
    print(
        f"Finished probability: {probability} with a Stixel-Score of {probab_score} % and a BBox-Score of {probab_bbox_score} % over {len(sample_results)} samples.")
    # wandb log
    f1_score = calculate_f1(precision=probab_score, recall=probab_bbox_score)
    logger.log({"F1-Score": f1_score, "probability": probability})
    logger.log({"precision": probab_score, "recall": probab_bbox_score})
    # Precision/ recall
    return {"probability": probability,
            "precision": probab_score,
            "recall": probab_bbox_score}


def calculate_f1(precision: float, recall: float):
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


if __name__ == "__main__":
    main()
