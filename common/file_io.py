# import time
# import random
# import string

# import os
# import pathlib
# import torch
# import urllib.request
# import gradio as gr
# from tqdm import tqdm
# import gc

# from tsr.system import TSR

# from modules.paths import models_path
# from modules.paths_internal import default_output_dir
# from modules import shared
# from omegaconf import OmegaConf


# # Device determination logic
# device = "cuda:0" if torch.cuda.is_available() else "cpu"


# def generate_random_filename(extension=".txt"):
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
#     filename = f"{timestamp}-{random_string}{extension}"
#     return filename


# # Builds the directory for the given model
# def models_dir_builder(folder_name):
#     models_dir = os.path.join(models_path, folder_name)
#     return models_dir


# # Function to download a file with a progress bar
# def download_with_progress(url, output_path):
#     response = urllib.request.urlopen(url)
#     total = int(response.headers.get('content-length', 0))
#     with tqdm(total=total, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(output_path)}") as bar:
#         with open(output_path, 'wb') as f:
#             while True:
#                 data = response.read(8192)
#                 if not data:
#                     break
#                 f.write(data)
#                 bar.update(len(data))


# # Function to ensure both model and config files are downloaded
# def download_model_and_config_if_needed(model_url, config_url, model_path, config_path):
#     if not os.path.exists(model_path):
#         download_with_progress(model_url, model_path)
#     if not os.path.exists(config_path):
#         download_with_progress(config_url, config_path)


# def unload_model(model):
#     # Delete the model
#     del model
#     # Clear the CUDA cache (if using GPU)
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     # Manually trigger garbage collection
#     gc.collect()