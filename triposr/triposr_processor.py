import torch
import gradio as gr
import os
import pathlib

from tsr import TSR
from utils import remove_background, resize_foreground, to_gradio_3d_orientation

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


model_root = os.path.join(models_path, 'TripoSR')
os.makedirs(model_root, exist_ok=True)
triposr_model_filenames = []

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