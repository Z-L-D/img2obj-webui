import torch
import gradio as gr
import os
import pathlib

from modules import script_callbacks
from modules.paths import models_path
# from modules.paths import output_path  # Assuming output_path is correctly imported
from modules.paths_internal import default_output_dir
from modules.ui_common import ToolButton, refresh_symbol
from modules.ui_components import ResizeHandleRow
from modules import shared

from modules_forge.forge_util import numpy_to_pytorch, pytorch_to_numpy
from ldm_patched.modules.sd import load_checkpoint_guess_config
from ldm_patched.modules import model_management

import tempfile
import time
import random
import string

import numpy as np
import rembg
from PIL import Image

from triposr.tsr import TSR
from triposr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


model_root = os.path.join(models_path, 'TripoSR')
os.makedirs(model_root, exist_ok=True)
triposr_model_filenames = []

def get_rembg_model_choices():
    # List of available models. 
    return [
        "dis_anime",
        "dis_general_use",
        "silueta",
        "u2net_cloth_seg", 
        "u2net_human_seg", 
        "u2net", 
        "u2netp", 
    ]
        # "sam", #- FIXME - not currently working

def update_model_filenames():
    global triposr_model_filenames
    triposr_model_filenames = [
        pathlib.Path(x).name for x in
        shared.walk_files(model_root, allowed_extensions=[".pt", ".ckpt", ".safetensors"])
    ]
    return triposr_model_filenames

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

# rembg_session = rembg.new_session(model_name="dis_general_use")

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")
    
def check_cutout_image(processed_image):
    if processed_image is None:
        raise gr.Error("No cutout image uploaded!")


def preprocess(
    input_image, 
    rembg_model,
    do_remove_background, 
    foreground_ratio,
    alpha_matting=False,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_size=0
):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(
            image,
            rembg.new_session(model_name=rembg_model),
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size
        )
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image

def generate_random_filename(extension=".txt"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    filename = f"{timestamp}-{random_string}{extension}"
    return filename

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

def generate(image, resolution, threshold):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, resolution=int(resolution), threshold=float(threshold))[0]
    mesh = to_gradio_3d_orientation(mesh)
    
    # Convert the mesh to a string or use a method to directly get the OBJ data
    obj_data = mesh.export(file_type='obj')  # This line might need adjustment based on how your mesh object works

    # Now save using the new function
    mesh_path = write_obj_to_triposr(obj_data)  # You could specify a filename if you want

    # Extract just the filename from the path
    filename = os.path.basename(mesh_path)

    relative_mesh_path = "output/TripoSR/" + filename

    return mesh_path, relative_mesh_path