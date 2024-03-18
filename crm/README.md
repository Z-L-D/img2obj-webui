# Convolutional Reconstruction Model

Official implementation for *CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model*.

## [Project Page](https://ml.cs.tsinghua.edu.cn/~zhengyi/CRM/) | [Arxiv](https://arxiv.org/abs/2403.05034) | [HF-Demo](https://huggingface.co/spaces/Zhengyi/CRM) | [Weights](https://huggingface.co/Zhengyi/CRM)

https://github.com/thu-ml/CRM/assets/40787266/1792d7e3-3b37-486c-ac7e-b181c67166dd

## Try CRM 🍻
* Try CRM at [Huggingface Demo](https://huggingface.co/spaces/Zhengyi/CRM).
* Try CRM at [Replicate Demo](https://replicate.com/camenduru/crm). Thanks [@camenduru](https://github.com/camenduru)! 

## Install

Required packages are listed in `requirements.txt`.

## Inference

We suggest gradio for a visualized inference.

```
gradio app.py
```

![1710081811132](https://github.com/thu-ml/CRM/assets/40787266/04c5c503-6abc-408e-91f4-5d95cfdd41ab)

## Acknowledgement
- [ImageDream](https://github.com/bytedance/ImageDream)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [kiuikit](https://github.com/ashawkey/kiuikit)
- [GET3D](https://github.com/nv-tlabs/GET3D)

## Citation

```
@article{wang2024crm,
  title={CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model},
  author={Zhengyi Wang and Yikai Wang and Yifei Chen and Chendong Xiang and Shuo Chen and Dajiang Yu and Chongxuan Li and Hang Su and Jun Zhu},
  journal={arXiv preprint arXiv:2403.05034},
  year={2024}
}
```
