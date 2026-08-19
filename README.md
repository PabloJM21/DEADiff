# DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations (CVPR 2024)

<div align="center">

 <a href='https://arxiv.org/abs/2403.06951'><img src='https://img.shields.io/badge/arXiv-2403.06951-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://tianhao-qi.github.io/DEADiff/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


_**[Tianhao Qi*](https://github.com/Tianhao-Qi/), [Shancheng Fang](https://tothebeginning.github.io/), [Yanze Wu✝](https://tothebeginning.github.io/), [Hongtao Xie✉](https://imcc.ustc.edu.cn/_upload/tpl/0d/13/3347/template3347/xiehongtao.html), [Jiawei Liu](https://scholar.google.com/citations?user=X21Fz-EAAAAJ&hl=en&authuser=1), <br>[Lang Chen](https://scholar.google.com/citations?user=h5xex20AAAAJ&hl=zh-CN), [Qian He](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&authuser=1&user=9rWWCgUAAAAJ), [Yongdong Zhang](https://scholar.google.com.hk/citations?user=hxGs4ukAAAAJ&hl=zh-CN)**_
<br><br>
(*Works done during the internship at ByteDance, ✝Project Lead, ✉Corresponding author)

From University of Science and Technology of China and ByteDance.

</div>


## 🔆 Introduction

**TL;DR:** We propose DEADiff, a generic method facilitating the synthesis of novel images that embody the style of a given reference image and adhere to text prompts.  <br>


### ⭐⭐ Stylized Text-to-Image Generation.

<div align="center">
<img src=docs/showcase_img.png>
<p>Stylized text-to-image results. Resolution: 512 x 512. (Compressed)</p>
</div>

### ⭐⭐ Style Transfer.

<div align="center">
<img src=docs/showcase_controlnet.png>
<p>Style transfer results with 
  <a href="https://github.com/lllyasviel/ControlNet.git" target="_blank">ControlNet</a>.
</p>
</div>


## 📝 Changelog
- __[2024.4.3]__: 🔥🔥 Release the inference code and pretrained checkpoint.
- __[2024.3.5]__: 🔥🔥 Release the project page.


## ⏳ TODO
- [x] Release the inference code.
- [ ] Release training data.


## ⚙️ Setup

```bash
#Create Python 3.9 venv
python3.9 -m venv deadiff
source deadiff/bin/activate
pip install --upgrade pip
#Install PyTorch stack
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
#Install LAVIS dependency pins
pip uninstall -y gradio fastapi starlette pydantic pydantic-core
pip install pydantic==1.10.12
pip install fairscale==0.4.4 timm==0.4.12 opencv-python-headless==4.5.5.64
#Install spaCy + thin
pip install spacy==3.5.3 thinc==8.1.10
#Install LAVIS BLIP‑Diffusion
pip install git+https://github.com/salesforce/LAVIS.git@20230801-blip-diffusion-edit
#Fix albumentations build
pip install --upgrade pip setuptools wheel
pip install setuptools==65.5.0
pip install --no-build-isolation albumentations==0.4.3

pip install -r requirements.txt
pip install -e .

pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-python-headless==4.5.5.64

pip uninstall -y huggingface_hub
pip install huggingface_hub==0.14.1


```

## 💫 Inference

1) Download the pretrained model from [Hugging Face](https://huggingface.co/qth/DEADiff/tree/main) and put it under ./pretrained/.
2) Run the commands in terminal.
```python3
python3 scripts/app.py
```
The Gradio app allows you to transfer style from the reference image. Just try it for more details.

Prompt: "A curly-haired boy"
![p](https://github.com/Tianhao-Qi/DEADiff_code_private/assets/37017794/bc0ebbf5-9bc9-4397-a0f6-dc291527571d)

Prompt: "A robot"
![p](https://github.com/Tianhao-Qi/DEADiff_code_private/assets/37017794/4b7bb264-aabb-42ae-bdc3-c20ebae5c0e6)

Prompt: "A motorcycle"
![p](https://github.com/Tianhao-Qi/DEADiff_code_private/assets/37017794/f23f8c4f-b72e-463c-9855-9767941e4932)

### ➕ Style Transfer with ControlNet

We support **style transfer with structural control** by combining DEADiff with [ControlNet](https://github.com/lllyasviel/ControlNet). This enables users to guide the spatial layout (e.g., edges or depth maps) of the generated images, while transferring the visual style from a reference image.

To perform style transfer with ControlNet, please download the following pretrained models:
- `control_sd15_canny.pth`: [Download](https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth) → place it under `./pretrained/`
- `control_sd15_depth.pth`: [Download](https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth) → place it under `./pretrained/`
- `dpt_hybrid-midas-501f0c75.pt` (for depth estimation): [Download](https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt) → place it under `ldm/controlnet/annotator/ckpts/`
These checkpoints are required for Canny and Depth-based ControlNet stylization modes.
Then run the following commands in terminal.
```python3
# Canny-based control
python3 scripts/app_canny_control.py
```

For non-interactive cluster jobs (no UI), use the batch CLI:
```bash
python3 scripts/batch_canny_control.py \
  --style_image /path/to/style.jpg \
  --content_image /path/to/content.jpg \
  --canny_map /path/to/canny_map.png \
  --prompt "best quality, extremely detailed, a futuristic city" \
  --output_dir outputs/canny_batch_run \
  --batch_size 1 \
  --ddim_steps 40 \
  --save_canny_map
```
This command saves generated images to `--output_dir` and exits, which makes it suitable for schedulers such as Slurm.
If `--canny_map` is provided, that map is used as the structural control signal instead of auto-extracting edges from `--content_image`.

```python3
# Depth-based control
python3 scripts/app_depth_control.py
```

## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****

## ✈️ Citation

```bibtex
@article{qi2024deadiff,
  title={DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations},
  author={Qi, Tianhao and Fang, Shancheng and Wu, Yanze and Xie, Hongtao and Liu, Jiawei and Chen, Lang and He, Qian and Zhang, Yongdong},
  journal={arXiv preprint arXiv:2403.06951},
  year={2024}
}
```

## 📭 Contact
If your have any comments or questions, feel free to contact [qth@mail.ustc.edu.cn](qth@mail.ustc.edu.cn)