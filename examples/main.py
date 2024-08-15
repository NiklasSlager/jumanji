# @title Set up JAX for available hardware (run me) { display-mode: "form" }

import subprocess
import os

# Based on https://stackoverflow.com/questions/67504079/how-to-check-if-an-nvidia-gpu-is-available-on-my-system
try:
    subprocess.check_output('nvidia-smi')
    print("a GPU is connected.")
except Exception:
    # TPU or CPU
    if "COLAB_TPU_ADDR" in os.environ and os.environ["COLAB_TPU_ADDR"]:
        import jax.tools.colab_tpu

        jax.tools.colab_tpu.setup_tpu()
        print("A TPU is connected.")
    else:
        print("Only CPU accelerator is connected.")
Only CPU accelerator is connected.
import warnings
warnings.filterwarnings("ignore")

from jumanji.training.train import train
from hydra import compose, initialize
env = "distillation"  
agent = "a2c"  # @param ['random', 'a2c']
#@title Download Jumanji Configs (run me) { display-mode: "form" }

import os
import requests


def download_file(url: str, file_path: str) -> None:
    # Send an HTTP GET request to the URL
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        print("Failed to download the file.")


os.makedirs("configs", exist_ok=True)
config_url = "https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/config.yaml"
download_file(config_url, "configs/config.yaml")
env_url = f"https://raw.githubusercontent.com/instadeepai/jumanji/main/jumanji/training/configs/env/{env}.yaml"
os.makedirs("configs/env", exist_ok=True)
download_file(env_url, f"configs/env/{env}.yaml")
with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name="config.yaml", overrides=[f"env={env}", f"agent={agent}", "logger.type=tensorboard", "logger.save_checkpoint=true"])

train(cfg)
