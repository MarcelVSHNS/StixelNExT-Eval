import os.path

import stixel as stx
import yaml
from PIL import Image

import wandb

if __name__ == "__main__":
    with open('config.yaml') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    """
    loader = WaymoDataLoader(data_dir=config['metric_data_path'],
                             first_only=True)
    idx = random.randint(0, len(loader) - 1)
    print(f"Random index: {idx}")
    drive = loader[idx]
    sample: WaymoData = drive[0]
    """
    run = wandb.init(project="StixelNExT-Pro",
                     job_type="analysis",
                     tags=["analysis"]
                     )
    artifact = run.use_artifact(f"{config['artifact']}", type='model')
    model_cfg = artifact.metadata
    path = os.path.join("/home/marcel/workspace/StixelNExT-Evaluation/models", artifact.digest)
    os.makedirs(path, exist_ok=True)
    artifact.download(path)

    analysis_artifact = wandb.Artifact('model_analysis', type='model', description="Analyzed model artifact")
    analysis_artifact.metadata = {
        "optimal_thres": 0.48,
        "blaa": 'hallo'
    }
    run.log({"test": 4.5})
    
    stx_img: Image = stx.draw_stixels_on_image(sample.stxl_wrld)
    path = "test.png"
    stx_img.save(path)
    analysis_artifact.add_file(path)
    run.log_artifact(analysis_artifact)
    run.finish()

    """
    tst = sample.stxl_wrld.stixel[0]
    results, stixel_pts, stixel_colors = evaluate_sample_3dbbox(sample.stxl_wrld, sample.bboxes)
    stxl_img = stx.draw_stixels_on_image(stxl_wrld=sample.stxl_wrld)
    stxl_img.show()
    print(results)
    # Visualise point cloud
    draw_stixel_and_bboxes(stixel_pts, stixel_colors, sample.bboxes)
    """
