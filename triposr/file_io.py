import os
import pathlib
import torch
import gradio as gr

from common.common import generate_random_filename
from triposr.tsr import TSR

from modules.paths import models_path
from modules.paths_internal import default_output_dir
from modules import shared

# if torch.cuda.is_available():
#     device = "cuda:0"
# else:
#     device = "cpu"

triposr_model_filenames = []

def update_model_filenames():
    global triposr_model_filenames
    model_root = os.path.join(models_path, 'TripoSR')
    os.makedirs(model_root, exist_ok=True)
    triposr_model_filenames = [
        pathlib.Path(x).name for x in
        shared.walk_files(model_root, allowed_extensions=[".pt", ".ckpt", ".safetensors"])
    ]
    return triposr_model_filenames

# model = TSR.from_pretrained(
#     "stabilityai/TripoSR",
#     config_name="config.yaml",
#     weight_name="model.ckpt",
# )

# adjust the chunk size to balance between speed and memory usage
# model.renderer.set_chunk_size(8192)
# model.to(device)

def check_cutout_image(processed_image):
    if processed_image is None:
        raise gr.Error("No cutout image uploaded!")

def write_obj_to_triposr(obj_data, filename=None):
    triposr_folder = os.path.join(default_output_dir, 'TripoSR')
    os.makedirs(triposr_folder, exist_ok=True)  # Ensure the directory exists

    if filename is None:
        filename = generate_random_filename('.obj')  # Implement or use an existing function to generate a unique filename

    full_path = os.path.join(triposr_folder, filename)

    # Assuming obj_data is a string containing the OBJ file data
    with open(full_path, 'w') as file:
        file.write(obj_data)

    return full_path