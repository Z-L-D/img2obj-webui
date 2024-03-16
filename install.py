# install.py
import subprocess
import os
import sys
from typing import Any
import pkg_resources
from tqdm import tqdm
import urllib.request
from packaging import version as pv

# Adjust these imports based on your project structure
from triposr.file_io import download_model_and_config_if_needed, load_model_on_device, models_dir

# Current version of your extension
current_version = '1.0'

# Assuming models_path is correctly imported or defined
try:
    from modules.paths_internal import models_path
except ImportError:
    try:
        from modules.paths import models_path
    except ImportError:
        models_path = os.path.abspath("models")

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
req_file = os.path.join(BASE_PATH, "requirements.txt")

# Define model and config URLs
triposr_model_url = "https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt"
triposr_config_url = "https://huggingface.co/stabilityai/TripoSR/resolve/main/config.yaml"
# Define model and config paths
triposr_model_path = os.path.join(models_dir, "model.ckpt")
triposr_config_path = os.path.join(models_dir, "config.yaml")

# Device selection logic
# device = "cuda:0" if torch.cuda.is_available() else "cpu"

installation_marker = os.path.join(BASE_PATH, ".install_complete")

def pip_install(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args], check=True)

def is_installed(package: str, version: str | None = None, strict: bool = True):
    try:
        has_package = pkg_resources.get_distribution(package)
        if has_package is not None:
            installed_version = has_package.version
            if (strict and installed_version != version) or (not strict and pv.parse(installed_version) < pv.parse(version)):
                return False
            else:
                return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")

needs_installation = True
if os.path.exists(installation_marker):
    with open(installation_marker, 'r') as f:
        installed_version = f.read().strip()
    if installed_version == current_version:
        needs_installation = False

if needs_installation:
    print("Installation or update needed. Proceeding...")

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Download the TripoSR model and config files if they don't exist
    download_model_and_config_if_needed(triposr_model_url, triposr_config_url, triposr_model_path, triposr_config_path)
    
    # Load the model to verify it's correctly set up
    # Note: If the model loading step is not needed during installation, you can remove this
    # model = load_model_on_device(triposr_config_path, triposr_model_path, device)
    
    # Install required packages
    with open(req_file) as file:
        for package in file:
            package_name, _, package_version = package.strip().partition("==")
            if not is_installed(package_name, package_version, strict=True):
                pip_install(package.strip())

    # After successful installation/update, write the current version to the marker file
    with open(installation_marker, 'w') as f:
        f.write(current_version)
    
    print("Installation or update complete.")
else:
    print("No installation or update needed. Launching with the current setup...")
