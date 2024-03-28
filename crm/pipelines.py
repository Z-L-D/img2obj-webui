# import torch
# from crm.libs.base_utils import do_resize_content
# from crm.imagedream.ldm.util import (
#     instantiate_from_config,
#     get_obj_from_str,
# )
# from omegaconf import OmegaConf
# from PIL import Image
# import numpy as np


# class TwoStagePipeline(object):
#     def __init__(
#         self,
#         stage1_model_config,
#         stage2_model_config,
#         stage1_sampler_config,
#         stage2_sampler_config,
#         device="cuda",
#         dtype=torch.float16,
#         resize_rate=1,
#     ) -> None:
#         """
#         only for two stage generate process.
#         - the first stage was condition on single pixel image, gererate multi-view pixel image, based on the v2pp config
#         - the second stage was condition on multiview pixel image generated by the first stage, generate the final image, based on the stage2-test config
#         """
#         self.resize_rate = resize_rate

#         self.stage1_model = instantiate_from_config(OmegaConf.load(stage1_model_config.config).model)
#         self.stage1_model.load_state_dict(torch.load(stage1_model_config.resume, map_location="cpu"), strict=False)
#         self.stage1_model = self.stage1_model.to(device).to(dtype)

#         self.stage2_model = instantiate_from_config(OmegaConf.load(stage2_model_config.config).model)
#         sd = torch.load(stage2_model_config.resume, map_location="cpu")
#         self.stage2_model.load_state_dict(sd, strict=False)
#         self.stage2_model = self.stage2_model.to(device).to(dtype)

#         self.stage1_model.device = device
#         self.stage2_model.device = device
#         self.device = device
#         self.dtype = dtype
#         self.stage1_sampler = get_obj_from_str(stage1_sampler_config.target)(
#             self.stage1_model, device=device, dtype=dtype, **stage1_sampler_config.params
#         )
#         self.stage2_sampler = get_obj_from_str(stage2_sampler_config.target)(
#             self.stage2_model, device=device, dtype=dtype, **stage2_sampler_config.params
#         )

#     def stage1_sample(
#         self,
#         pixel_img,
#         prompt="3D assets",
#         neg_texts="uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear.",
#         step=50,
#         scale=5,
#         ddim_eta=0.0,
#     ):
#         if type(pixel_img) == str:
#             pixel_img = Image.open(pixel_img)

#         if isinstance(pixel_img, Image.Image):
#             if pixel_img.mode == "RGBA":
#                 background = Image.new('RGBA', pixel_img.size, (0, 0, 0, 0))
#                 pixel_img = Image.alpha_composite(background, pixel_img).convert("RGB")
#             else:
#                 pixel_img = pixel_img.convert("RGB")
#         else:
#             raise
#         uc = self.stage1_sampler.model.get_learned_conditioning([neg_texts]).to(self.device)
#         stage1_images = self.stage1_sampler.i2i(
#             self.stage1_sampler.model,
#             self.stage1_sampler.size,
#             prompt,
#             uc=uc,
#             sampler=self.stage1_sampler.sampler,
#             ip=pixel_img,
#             step=step,
#             scale=scale,
#             batch_size=self.stage1_sampler.batch_size,
#             ddim_eta=ddim_eta,
#             dtype=self.stage1_sampler.dtype,
#             device=self.stage1_sampler.device,
#             camera=self.stage1_sampler.camera,
#             num_frames=self.stage1_sampler.num_frames,
#             pixel_control=(self.stage1_sampler.mode == "pixel"),
#             transform=self.stage1_sampler.image_transform,
#             offset_noise=self.stage1_sampler.offset_noise,
#         )

#         stage1_images = [Image.fromarray(img) for img in stage1_images]
#         stage1_images.pop(self.stage1_sampler.ref_position)
#         return stage1_images

#     def stage2_sample(self, pixel_img, stage1_images, scale=5, step=50):
#         if type(pixel_img) == str:
#             pixel_img = Image.open(pixel_img)

#         if isinstance(pixel_img, Image.Image):
#             if pixel_img.mode == "RGBA":
#                 background = Image.new('RGBA', pixel_img.size, (0, 0, 0, 0))
#                 pixel_img = Image.alpha_composite(background, pixel_img).convert("RGB")
#             else:
#                 pixel_img = pixel_img.convert("RGB")
#         else:
#             raise
#         stage2_images = self.stage2_sampler.i2iStage2(
#             self.stage2_sampler.model,
#             self.stage2_sampler.size,
#             "3D assets",
#             self.stage2_sampler.uc,
#             self.stage2_sampler.sampler,
#             pixel_images=stage1_images,
#             ip=pixel_img,
#             step=step,
#             scale=scale,
#             batch_size=self.stage2_sampler.batch_size,
#             ddim_eta=0.0,
#             dtype=self.stage2_sampler.dtype,
#             device=self.stage2_sampler.device,
#             camera=self.stage2_sampler.camera,
#             num_frames=self.stage2_sampler.num_frames,
#             pixel_control=(self.stage2_sampler.mode == "pixel"),
#             transform=self.stage2_sampler.image_transform,
#             offset_noise=self.stage2_sampler.offset_noise,
#         )
#         stage2_images = [Image.fromarray(img) for img in stage2_images]
#         return stage2_images

#     def set_seed(self, seed):
#         self.stage1_sampler.seed = seed
#         self.stage2_sampler.seed = seed

#     def __call__(self, pixel_img, prompt="3D assets", scale=5, step=50):
#         pixel_img = do_resize_content(pixel_img, self.resize_rate)
#         stage1_images = self.stage1_sample(pixel_img, prompt, scale=scale, step=step)
#         stage2_images = self.stage2_sample(pixel_img, stage1_images, scale=scale, step=step)

#         return {
#             "ref_img": pixel_img,
#             "stage1_images": stage1_images,
#             "stage2_images": stage2_images,
#         }


# if __name__ == "__main__":

#     stage1_config = OmegaConf.load("configs/nf7_v3_SNR_rd_size_stroke.yaml").config
#     stage2_config = OmegaConf.load("configs/stage2-v2-snr.yaml").config
#     stage2_sampler_config = stage2_config.sampler
#     stage1_sampler_config = stage1_config.sampler

#     stage1_model_config = stage1_config.models
#     stage2_model_config = stage2_config.models

#     pipeline = TwoStagePipeline(
#         stage1_model_config,
#         stage2_model_config,
#         stage1_sampler_config,
#         stage2_sampler_config,
#     )

#     img = Image.open("assets/astronaut.png")
#     rt_dict = pipeline(img)
#     stage1_images = rt_dict["stage1_images"]
#     stage2_images = rt_dict["stage2_images"]
#     np_imgs = np.concatenate(stage1_images, 1)
#     np_xyzs = np.concatenate(stage2_images, 1)
#     Image.fromarray(np_imgs).save("pixel_images.png")
#     Image.fromarray(np_xyzs).save("xyz_images.png")
