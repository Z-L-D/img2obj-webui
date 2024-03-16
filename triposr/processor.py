import torch
import gradio as gr
import os
import pathlib

# from triposr.file_io import device, model, write_obj_to_triposr
from triposr.utils import to_gradio_3d_orientation
from triposr.file_io import load_model_on_device, write_obj_to_triposr, models_dir

from modules.paths import models_path
from modules.paths_internal import default_output_dir

# def generate(model_name, image, resolution, threshold):
#     # Determine model and config paths based on model_name
#     model_path = os.path.join(models_dir, f"{model_name}.ckpt")  # Example path, adjust as needed
#     config_path = os.path.join(models_dir, f"config.yaml")  # Example path, adjust as needed

#     # Load model using modified load_model_on_device function
#     model, device = load_model_on_device(config_path, model_path)

#     if image.mode == 'RGBA':
#         image = image.convert('RGB')
#     scene_codes = model(image, device=device)
#     mesh = model.extract_mesh(scene_codes, resolution=int(resolution), threshold=float(threshold))[0]
#     mesh = to_gradio_3d_orientation(mesh)
    
#     # Convert the mesh to a string or use a method to directly get the OBJ data
#     obj_data = mesh.export(file_type='obj')  # This line might need adjustment based on how your mesh object works

#     # Now save using the new function
#     mesh_path = write_obj_to_triposr(obj_data)  # You could specify a filename if you want

#     # Extract just the filename from the path
#     filename = os.path.basename(mesh_path)

#     relative_mesh_path = "output/TripoSR/" + filename

#     return mesh_path, relative_mesh_path

def generate_pipeline(model_name, image, resolution, threshold):
    print("\n")
    print("=====================================================")
    print("TripoSR Generation started.")
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("model_name: " + model_name)
    print("resolution: " + str(resolution))
    print("threshold: " + str(threshold))
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("=====================================================")
    print("\n")


def generate(model_name, image, resolution, threshold):
    # Determine model and config paths based on model_name
    model_path = os.path.join(models_dir, f"{model_name}.ckpt")  # Example path, adjust as needed
    config_path = os.path.join(models_dir, f"config.yaml")  # Example path, adjust as needed

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