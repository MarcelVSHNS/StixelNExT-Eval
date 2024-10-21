""" Evaluate, model + weights
"""
import os.path
from datetime import datetime, timedelta
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml
from tqdm import tqdm

import wandb
from dataloader import WaymoDataLoader, StixelModel
from metric import evaluate_sample_3dbbox

overall_start_time = datetime.now()
os.environ["WANDB_REPORT_API_ENABLE_V2"] = "True"
os.environ["WANDB_REPORT_API_DISABLE_MESSAGE"] = "True"

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)


def main():
    mp.set_start_method('spawn')
    loader = WaymoDataLoader(data_dir=config['metric_data_path'],
                             first_only=False)
    # create logger
    logger = wandb.init(project="StixelNExT-Pro",
                        job_type="analysis",
                        tags=["analysis"]
                        )
    artifact = logger.use_artifact(f"{config['artifact']}", type='model')
    # create model
    if config["device"] == "gpu":
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device('cpu')

    with mp.Manager() as manager:
        stxl_model = StixelModel(device=dev)
        stxl_model.model.share_memory()
        stxl_model.info()
        gpu_lock = manager.Lock()
        # create results_folder
        result_dir = os.path.join('results', stxl_model.checkpoint_name)
        os.makedirs(result_dir, exist_ok=True)

        # multiprocess every probability
        probabilities = np.arange(config["from"], config["to"], config["in"])
        evaluate_partial = partial(evaluate,
                                   dataloader=loader,
                                   model=stxl_model,
                                   gpu_lock=gpu_lock
                                   )
        start_time = datetime.now()
        results = []

        with mp.Pool() as pool:
            with tqdm(total=len(probabilities), desc="Progress", dynamic_ncols=True, position=0, leave=True) as pbar:
                for result, progress_info in pool.imap(evaluate_partial, probabilities):
                    pbar.set_description(
                        f"Times - Inference: {progress_info[0]}, Evaluation: {progress_info[1]} - {datetime.now() - start_time}")
                    results.append(result)
                    pbar.update(1)
            # results = pool.map(analyse_partial, index_list)
        print(f"Finished in {datetime.now() - start_time}.")

    # organize results in dict
    probabilities = []
    precisions = []
    recalls = []
    for result in results:
        probabilities.append(result['probability'])
        precisions.append(result['precision'])
        recalls.append(result['recall'])
        logger.log({
            "Precision": result['precision'],
            "Recall": result['recall'],
            "F1_score": result['F1-Score'],
            "Probability": result['probability']
        })
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
             model: StixelModel,
             gpu_lock: mp.Lock
             ):
    probab_result = {'Stixel-Score': np.array([]), 'BBox-Score': np.array([])}
    sample_results = {}
    stxl_model = model
    result_dir = os.path.join('results', stxl_model.checkpoint_name)
    os.makedirs(result_dir, exist_ok=True)
    times = [[], []]
    index = 1
    for record in dataloader:
        sample_idx = 0
        # print(f"Starting record with {len(record)} samples.")
        for sample in record:
            start_time = datetime.now()
            # Inference a Stixel World
            start_inf = datetime.now()
            with gpu_lock:
                stxl_infer = stxl_model.inference(sample.image)
                torch.cuda.empty_cache()
            stxl_wrld = stxl_model.revert(stxl_infer, probability=probability, calib=sample.calib)
            if len(stxl_wrld.stixel) > config['stx_dropout_threshold'] and config['stx_dropout']:
                continue
            times[0].append(datetime.now() - start_inf)
            # Apply the evaluation
            start_eval = datetime.now()
            results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(stxl_wrld, sample.bboxes)
            times[1].append(datetime.now() - start_eval)
            # print(f"Evaluation: {datetime.now() - start_eval}")
            probab_result['Stixel-Score'] = np.append(probab_result['Stixel-Score'], results['Stixel-Score'])
            probab_result['BBox-Score'] = np.append(probab_result['BBox-Score'], results['BBox-Score'])
            sample_results[sample.name] = results
            sample_time = datetime.now() - start_time
            results_short = results.copy()
            results_short.pop('bbox_dist', None)
            # print(f"{sample.name} (idx={sample_idx}) with Stixel-Score:{results['Stixel-Score']} and BBox-Score: {results['BBox-Score']} within {sample_time}. {results_short}")
            sample_idx += 1
        step_time = datetime.now() - overall_start_time
        # print("#####################################################################")
        # print(f"Record-file {index}/ {len(dataloader)} evaluated with [Stixel-Score: {np.mean(probab_result['Stixel-Score'])} %/ BBox-Score of {np.mean(probab_result['BBox-Score'])} %]. Time elapsed: {step_time}")
        index += 1
    probab_score = np.mean(probab_result['Stixel-Score'])
    probab_bbox_score = np.mean(probab_result['BBox-Score'])

    df = pd.DataFrame.from_dict(sample_results, orient='index')
    df.index.name = 'Sample_ID'
    df.to_csv(os.path.join(result_dir,
                           f"{dataloader.name}-{config['results_name']}_PROB-{probability}_StixelScore-{probab_score}_bboxScore-{probab_bbox_score}.csv"))
    # print(f"Finished probability: {probability} with a Stixel-Score of {probab_score} % and a BBox-Score of {probab_bbox_score} % over {len(sample_results)} samples.")
    # wandb log
    f1_score = calculate_f1(precision=probab_score, recall=probab_bbox_score)
    average_times = (
        sum(times[0], timedelta()) / len(times[0]),
        sum(times[1], timedelta()) / len(times[1])
    )
    # Precision/ recall
    return {"probability": probability,
            "precision": probab_score,
            "recall": probab_bbox_score,
            "F1-Score": f1_score}, average_times


def calculate_f1(precision: float, recall: float):
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


if __name__ == "__main__":
    main()
