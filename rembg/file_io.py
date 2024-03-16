import gradio as gr

rembg_model_filenames = []
def update_rembg_model_filenames():
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

def check_input_image(input_image):
    print("img2obj: Checking for preprocessor input image")
    if input_image is None:
        print("img2obj: No preprocessor image uploaded!")
        raise gr.Error("No preprocessor image uploaded!")