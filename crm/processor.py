import argparse
import numpy as np
import gradio as gr
from omegaconf import OmegaConf
import torch
from PIL import Image
import PIL
from huggingface_hub import hf_hub_download
import os
import rembg
from typing import Any
import json
import os
import json
import argparse

from crm.model import CRM
from crm.inference import generate3d
from crm.pipelines import TwoStagePipeline
from .file_io import models_dir

def crm_generate(model_name, image, resolution, threshold):
    # Determine model and config paths based on model_name
    model_path = os.path.join(models_dir, f"{model_name}")  # Example path, adjust as needed
    config_path = os.path.join(models_dir, f"config.yaml")  # Example path, adjust as needed