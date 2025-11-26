import os
import shutil

from pathlib import Path
from importlib.machinery import SourceFileLoader

from scripts.splatam import rgbd_slam
from utils.common_utils import seed_everything

def run_slam(config_path, queue):
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} does not exist.")
    
    experiment = SourceFileLoader(os.path.basename(config_path), config_path).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(experiment.config["workdir"], experiment.config["run_name"])
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(config_path, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config, queue)