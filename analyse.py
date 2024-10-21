# import multiprocessing
import os
import os.path
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import stixel as stx
import torch
import torch.multiprocessing as mp
import yaml
from stixel.utils.packing import add_image
from tqdm import tqdm

import wandb
from dataloader import WaymoDataLoader, StixelModel
from metric import evaluate_sample_3dbbox

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

overall_start_time = datetime.now()
os.environ["WANDB_REPORT_API_ENABLE_V2"] = "True"
os.environ["WANDB_REPORT_API_DISABLE_MESSAGE"] = "True"


def main():
    mp.set_start_method('spawn')
    loader = WaymoDataLoader(data_dir=config["metric_data_path"],
                             first_only=True)
    # logger
    logger = wandb.init(project="StixelNExT-Pro",
                        job_type="analysis",
                        tags=["analysis"]
                        )
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

        index_list = list(range(0, len(loader), 32))
        analyse_partial = partial(analyse,
                                  dataloader=loader,
                                  model=stxl_model,
                                  gpu_lock=gpu_lock
                                  )
        start_time = datetime.now()

        results = []
        with mp.Pool() as pool:
            with tqdm(total=len(index_list), desc="Progress", dynamic_ncols=True, position=0, leave=True) as pbar:
                for result, progress_info in pool.imap(analyse_partial, index_list):
                    pbar.set_description(
                        f"Times - Inference: {progress_info[0]}, Evaluation: {progress_info[1]} - {datetime.now() - start_time}")
                    results.append(result)
                    pbar.update(1)
            # results = pool.map(analyse_partial, index_list)
        print(f"Finished in {datetime.now() - start_time}.")

    # save every precision/ recall of every sample and probability
    # e.g. a dict with probability as key and a list of precisions (by sample) as value
    precision = defaultdict(list)
    recall = defaultdict(list)
    num_stx = defaultdict(list)
    for sample in results:
        for probab, result in sample['results'].items():
            precision[probab].append(result['precision'])
            recall[probab].append(result['recall'])
            num_stx[probab].append(len(result['stxl_wrld'].stixel))

    avg_precision = {probab: sum(vals) / len(vals) for probab, vals in precision.items()}
    avg_recall = {probab: sum(vals) / len(vals) for probab, vals in recall.items()}
    avg_num_stx = {probab: sum(vals) / len(vals) for probab, vals in num_stx.items()}

    for probab in avg_precision:
        wandb.log({
            "Precision": avg_precision[probab],
            "Recall": avg_recall[probab],
            "Num Stixel": avg_num_stx[probab],
            "Probability": probab
        })

    sorted_probabilities = sorted(avg_precision.keys())
    sorted_avg_precision = [avg_precision[probab] for probab in sorted_probabilities]
    sorted_avg_recall = [avg_recall[probab] for probab in sorted_probabilities]
    # draw precision recall curve
    plt.figure()
    plt.plot(sorted_avg_recall, sorted_avg_precision, label='NN', marker='o', color='turquoise')
    plt.plot(1.155, 0.974, label='GT', marker='x', color='fuchsia')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    name = f"{loader.name}-{config['results_name']} {stxl_model.checkpoint_name}"
    plt.title(name)
    plt.legend()
    plt.savefig(os.path.join(result_dir, name + '.png'))

    result_idx = 2
    sample = results[result_idx]
    image = sample['image']
    table = wandb.Table(columns=["sample_id", "probability", "image"])
    for probab, result in sample['results'].items():
        stxl_wrld = result['stxl_wrld']
        stxl_wrld = add_image(stxl_wrld, image)
        stxl_img = stx.draw_stixels_on_image(stxl_wrld)
        stxl_img_wandb = wandb.Image(stxl_img)
        table.add_data(f"sample_id{result_idx}", probab, stxl_img_wandb)
    wandb.log({"image_variants": table})
    logger.finish()
    """
    best_f1 = {"f1_score": -100000.0}
    for probab, result in sample['results'].items():
        if result['f1_score'] > best_f1["f1_score"]:
            best_f1["image"] = sample['image']
            best_f1["stxl_wrld"] = result['stxl_wrld']
    # show best in 3d and 2d
    stxl_wrld = add_image(best_f1["stxl_wrld"], best_f1["image"])
    stxl_img = stx.draw_stixels_on_image(stxl_wrld)
    stxl_img.show()
    stx.draw_stixels_in_3d(stxl_wrld)
    """


def analyse(sample_idx: int,
            dataloader: WaymoDataLoader,
            model: StixelModel,
            gpu_lock: mp.Lock):
    sample = dataloader[sample_idx][0]
    stxl_model = model
    result_dir = os.path.join('results', stxl_model.checkpoint_name)
    os.makedirs(result_dir, exist_ok=True)
    result_list = {}
    times = [[], []]
    if config["device"] == "gpu":
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device('cpu')
    stxl_model = StixelModel(device=dev)
    for probability in np.arange(0.0, 1.0, 0.05):
        # Inference a Stixel World
        start_inf = datetime.now()
        with gpu_lock:
            stxl_infer = stxl_model.inference(sample.image)
            torch.cuda.empty_cache()
        stxl_wrld = stxl_model.revert(stxl_infer, probability=probability, calib=sample.calib)
        times[0].append(datetime.now() - start_inf)
        # if len(stxl_wrld.stixel) > 2000:
        #     continue
        # print(f"Inference: {datetime.now() - start_inf}")
        # Apply the evaluation
        start_eval = datetime.now()
        results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(stxl_wrld, sample.bboxes)
        times[1].append(datetime.now() - start_eval)
        # print(f"Evaluation: {datetime.now() - start_eval}")
        result_list[probability] = {"precision": results['Stixel-Score'],
                                    "recall": results['BBox-Score'],
                                    "f1_score": calculate_f1(precision=results['Stixel-Score'],
                                                             recall=results['BBox-Score']),
                                    "results": results,
                                    "pts": stixel_pts,
                                    "colors": stixel_colors,
                                    "stxl_wrld": stxl_wrld}
        # results_short = results.copy()
        # results_short.pop('bbox_dist', None)
        # print(f"{sample.name} @ {probability} with Precision:{results['Stixel-Score']} and Recall: {results['BBox-Score']} within {sample_time}. {results_short}")
        # print("#####################################################################")

    highest_f1_score = -1
    best_probability = None
    for probability, values in result_list.items():
        if values["f1_score"] > highest_f1_score:
            highest_f1_score = values["f1_score"]
            best_probability = probability
    average_times = (
        sum(times[0], timedelta()) / len(times[0]),
        sum(times[1], timedelta()) / len(times[1])
    )
    return {"idx": sample_idx,
            "image": sample.image,
            "pick": best_probability,
            "results": result_list}, average_times


def calculate_f1(precision: float, recall: float):
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0


if __name__ == "__main__":
    main()
