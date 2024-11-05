import subprocess
import os
import warnings
from jumanji.training.train import train
from hydra import compose, initialize
import requests

warnings.filterwarnings("ignore")

env = "distillation"
agent = "a2c"  # @param ['random', 'a2c']


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
config_url = "https://raw.githubusercontent.com/NiklasSlager/jumanji/main/jumanji/training/configs/config.yaml"
download_file(config_url, "configs/config.yaml")
env_url = f"https://raw.githubusercontent.com/NiklasSlager/jumanji/main/jumanji/training/configs/env/{env}.yaml"
os.makedirs("configs/env", exist_ok=True)
download_file(env_url, f"configs/env/{env}.yaml")
with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name="config.yaml", overrides=[f"env={env}", f"agent={agent}", "logger.type=tensorboard", "logger.save_checkpoint=true"])

train(cfg)
