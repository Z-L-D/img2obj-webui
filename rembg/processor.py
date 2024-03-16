import numpy as np
import rembg
import PIL
import torch
import torch.nn.functional as F
from typing import Any, List, Union



class ImagePreprocessor:
    def convert_and_resize(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        size: int,
    ):
        if isinstance(image, PIL.Image.Image):
            image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = torch.from_numpy(image.astype(np.float32) / 255.0)
            else:
                image = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            pass

        batched = image.ndim == 4

        if not batched:
            image = image[None, ...]
        image = F.interpolate(
            image.permute(0, 3, 1, 2),
            (size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).permute(0, 2, 3, 1)
        if not batched:
            image = image[0]
        return image

    def __call__(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        size: int,
    ) -> Any:
        if isinstance(image, (np.ndarray, torch.FloatTensor)) and image.ndim == 4:
            image = self.convert_and_resize(image, size)
        else:
            if not isinstance(image, list):
                image = [image]
            image = [self.convert_and_resize(im, size) for im in image]
            image = torch.stack(image, dim=0)
        return image

def remove_background(
    image,
    rembg_session=None,
    bgcolor=None,
    alpha_matting=False,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_size=10,
    **rembg_kwargs
):
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(
            image,
            session=rembg_session,
            bgcolor=bgcolor,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
            **rembg_kwargs
        )
    return image


def resize_foreground(
    image: PIL.Image.Image,
    ratio: float,
) -> PIL.Image.Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = PIL.Image.fromarray(new_image)
    return new_image
    
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
    print("\n")
    print("=====================================================")
    print("REMBG Preprocess started.")
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("model_name: " + rembg_model)
    print("remove background: " + str(do_remove_background))
    print("foreground_ratio: " + str(foreground_ratio))
    print("alpha_matting: " + str(alpha_matting))
    print("alpha_matting_foreground_threshold: " + str(alpha_matting_foreground_threshold))
    print("alpha_matting_background_threshold: " + str(alpha_matting_background_threshold))
    print("alpha_matting_erode_size: " + str(alpha_matting_erode_size))

    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = PIL.Image.fromarray((image * 255.0).astype(np.uint8))
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

    print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("=====================================================")
    print("\n")
    
    return image