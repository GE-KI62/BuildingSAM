
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import typer
from typing import Optional
import modules.preprocess as preprocessing
import modules.train as training 
import modules.inference as inferencing 
import modules.postprocess as postprocessing 
from modules.config import Config

app = typer.Typer()

@app.command()
def preprocess(config_path: str = "config.yaml"):
    config = Config.from_yaml(config_path, sub="preprocessor")
    preprocessing.process(config.preprocessor)

@app.command()
def train(config_path: str = "config.yaml"):
    config = Config.from_yaml(config_path, sub=["training", "model"])
    training.train(config.training, config.model)

@app.command()
def infer(config_path: str = "config.yaml"):
    config = Config.from_yaml(config_path, sub=["inference", "model"])
    inferencing.infer(config.inference, config.model)

@app.command()
def postprocess(config_path: str = "config.yaml"):
    config = Config.from_yaml(config_path, sub="postprocessor")
    postprocessing.postprocess(config.postprocessor)


if __name__ == "__main__":
    app()