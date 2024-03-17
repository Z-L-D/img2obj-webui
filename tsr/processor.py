import torch
import gradio as gr
import os
import pathlib

from .utils import to_gradio_3d_orientation
from .file_io import write_obj_to_triposr, unload_model, models_dir
from .system import TSR

from modules.paths import models_path
from modules.paths_internal import default_output_dir

device = "cuda:0" if torch.cuda.is_available() else "cpu"    

def triposr_generate(model_name, image, resolution, threshold):
    # Determine model and config paths based on model_name
    model_path = os.path.join(models_dir, f"{model_name}")  # Example path, adjust as needed
    config_path = os.path.join(models_dir, f"config.yaml")  # Example path, adjust as needed

    model = TSR.from_pretrained(
        models_dir,
        config_path,
        model_path,
    )

    model.renderer.set_chunk_size(8192)
    model.to(device)

    print("\n")
    print("=====================================================")
    print("TripoSR Generation started.")
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("model_name: " + model_name)
    print("resolution: " + str(resolution))
    print("threshold: " + str(threshold))
    print("model_path: " + str(model_path))
    print("config_path: " + str(config_path))
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    print("device: " + device)
    
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, resolution=int(resolution), threshold=float(threshold))[0]
    mesh = to_gradio_3d_orientation(mesh)
    
    # Convert the mesh to a string or use a method to directly get the OBJ data
    obj_data = mesh.export(file_type='obj')  # This line might need adjustment based on how your mesh object works

    # Now save using the new function
    mesh_path = write_obj_to_triposr(obj_data)  # You could specify a filename if you want

    print("mesh_path: " + str(mesh_path))

    # Extract just the filename from the path
    filename = os.path.basename(mesh_path)

    print("filename: " + str(filename))

    relative_mesh_path = "output/TripoSR/" + filename

    print("relative_mesh_path: " + relative_mesh_path)

    unload_model(model)

    print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("=====================================================")
    print("\n")
    
    return mesh_path, relative_mesh_path