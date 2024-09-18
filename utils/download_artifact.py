import os

import wandb
import yaml

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

run = wandb.init(project="StixelNExT-Pro",
                 job_type="analysis",
                 tags=["analysis"]
                 )
artifact = run.use_artifact(f"{config['artifact']}:latest", type='model')

path = os.path.join("/home/marcel/workspace/StixelNExT-Evaluation/models", artifact.id)
artifact.download(path)

wandb.finish()
