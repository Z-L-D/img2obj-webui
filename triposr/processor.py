import torch
import gradio as gr
import os
import pathlib

# from triposr.file_io import device, model, write_obj_to_triposr
from triposr.utils import to_gradio_3d_orientation

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