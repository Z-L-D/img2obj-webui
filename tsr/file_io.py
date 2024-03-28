# # file_io.py
# import os
# import pathlib
# import torch
# import urllib.request
# import gradio as gr
# from tqdm import tqdm
# import gc

# from common.file_io import generate_random_filename, models_dir_builder
# from tsr.system import TSR

# from modules.paths import models_path
# from modules.paths_internal import default_output_dir
# from modules import shared
# from omegaconf import OmegaConf

# # # Device determination logic
# # device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # # Directory for the TripoSR models
# # models_dir = os.path.join(models_path, "TripoSR")

# # # Function to download a file with a progress bar
# # def download_with_progress(url, output_path):
# #     response = urllib.request.urlopen(url)
# #     total = int(response.headers.get('content-length', 0))
# #     with tqdm(total=total, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(output_path)}") as bar:
# #         with open(output_path, 'wb') as f:
# #             while True:
# #                 data = response.read(8192)
# #                 if not data:
# #                     break
# #                 f.write(data)
# #                 bar.update(len(data))

# # # Function to ensure both model and config files are downloaded
# # def download_model_and_config_if_needed(model_url, config_url, model_path, config_path):
# #     if not os.path.exists(model_path):
# #         download_with_progress(model_url, model_path)
# #     if not os.path.exists(config_path):
# #         download_with_progress(config_url, config_path)

# # def unload_model(model):
# #     # Delete the model
# #     del model
# #     # Clear the CUDA cache (if using GPU)
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #     # Manually trigger garbage collection
# #     gc.collect()

# models_dir = models_dir_builder("TripoSR")

# triposr_model_filenames = []
# def update_triposr_model_filenames():
#     global triposr_model_filenames
#     model_root = models_dir
#     os.makedirs(model_root, exist_ok=True)
#     triposr_model_filenames = [
#         pathlib.Path(x).name for x in
#         shared.walk_files(model_root, allowed_extensions=[".pt", ".ckpt", ".safetensors"])
#     ]
#     return triposr_model_filenames

# def check_cutout_image(processed_image):
#     if processed_image is None:
#         raise gr.Error("No cutout image uploaded!")

# def write_obj_to_triposr(obj_data, filename=None):
#     triposr_folder = os.path.join(default_output_dir, 'TripoSR')
#     os.makedirs(triposr_folder, exist_ok=True)  # Ensure the directory exists

#     if filename is None:
#         filename = generate_random_filename('.obj')

#     full_path = os.path.join(triposr_folder, filename)

#     # Assuming obj_data is a string containing the OBJ file data
#     with open(full_path, 'w') as file:
#         file.write(obj_data)

#     return full_path
