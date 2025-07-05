# Copyright (2024) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
import accelerate
from safetensors.torch import load_file
import gradio as gr

import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision import transforms as T
from ldm.controlnet.annotator.util import HWC3, resize_image
from ldm.controlnet.annotator.canny import CannyDetector


apply_canny = CannyDetector()

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    print('loading done')
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, img_weight):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        assert isinstance(cond, dict)
        c_crossattn = cond['c_crossattn']
        c_concat = cond['c_concat']
        uc_crossattn = uncond['c_crossattn']
        uc_concat = uncond['c_concat']
        cond_in = []
        if isinstance(uc_crossattn[0], list):
            for uc, c in zip(uc_crossattn, c_crossattn):
                cond_in_temp = []
                for c_tmp, uc_tmp in zip(c, uc):
                    if c_tmp is None:
                        cond_in_temp.append(None)
                    else:
                        cond_in_temp.append(torch.cat([uc_tmp, c_tmp]))
                cond_in.append(cond_in_temp)
        else:
            for c in c_crossattn:
                if isinstance(c, list):
                    cond_in_temp = []
                    for c_tmp, uc in zip(c, uc_crossattn):
                        cond_in_temp.append(torch.cat([uc, c_tmp]))
                    cond_in.append(cond_in_temp)
                else:
                    cond_in.append(torch.cat([uc_crossattn, c]))
        cond_in = {'c_crossattn': cond_in, 'c_concat': [torch.cat([uc_concat[0], c_concat[0]], 0)]}
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in, img_weight=img_weight).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class DEADiff_CannyControl(object):
    def __init__(self,
                 config,
                 ckpt,
                 ckpt_controlnet):
        config = OmegaConf.load(f"{config}")
        config.model.params.control_stage_config.params.ckpt_path = ckpt_controlnet
        self.model = load_model_from_config(config, f"{ckpt}")

        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    def generate(self, prompt, image_style_input, image_content_input, subject_text, batch_size, ddim, ddim_steps, scale, img_weight, seed):
        accelerator = accelerate.Accelerator()
        device = accelerator.device
        seed_everything(seed)

        n_rows = 2 if batch_size >= 2 else 1
        if batch_size < 2:
            n_rows = 1
        elif batch_size < 5:
            n_rows = 2
        else:
            n_rows = 3
        prompts = batch_size * [prompt]

        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    uc = None
                    if scale != 1.0:
                        uc_encoder_hidden_states = self.model.get_learned_conditioning({'target_text':batch_size * ["over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"], 'subject_text': subject_text})
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    if subject_text == "style & content":
                        subject_text = ["style", "content"]
                    if subject_text == "None":
                        subject_text = None

                    ####### depth control #######
                    img = resize_image(HWC3(image_content_input), 384)
                    H, W, C = img.shape

                    detected_map = apply_canny(img, 100, 200)
                    detected_map = HWC3(detected_map)
                    canny_map = Image.fromarray(detected_map)

                    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                    control = torch.stack([control for _ in range(batch_size)], dim=0)
                    control = rearrange(control, 'b h w c -> b c h w').clone()

                    ####### conditioning #######
                    c_encoder_hidden_states = self.model.get_learned_conditioning({
                            'target_text':prompts,
                            'inp_image': 2*(T.ToTensor()(Image.fromarray(image_style_input).convert('RGB').resize((224, 224)))-0.5).unsqueeze(0).repeat(batch_size, 1,1,1).to('cuda'),
                            'subject_text': [subject_text]*batch_size,
                        })
                    uc, c = uc_encoder_hidden_states, c_encoder_hidden_states
                    cond = {"c_concat": [control], "c_crossattn": c}
                    un_cond = {"c_concat": [control], "c_crossattn": [uc, uc]}
                    
                    shape = [4, H // 8, W // 8]

                    if ddim == 'ddim':
                        sampler = DDIMSampler(self.model)
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=cond,
                                                         batch_size=batch_size,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=un_cond,
                                                         img_weight=img_weight)
                    else:
                        sigmas = self.model_wrap.get_sigmas(ddim_steps)
                        x = torch.randn([batch_size, *shape], device=device) * sigmas[0] # for GPU draw
                        
                        extra_args = {'cond': cond, 'uncond': un_cond, 'cond_scale': scale, 'img_weight': img_weight}
                        samples_ddim = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = accelerator.gather(x_samples_ddim)
                    
                    if accelerator.is_main_process:
                        all_samples = [T.ToPILImage()(x_sample_ddim) for x_sample_ddim in x_samples_ddim]
                        grid = make_grid(x_samples_ddim, nrow=n_rows)
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        all_samples.append(Image.fromarray(grid.astype(np.uint8)))
        return all_samples


def interface(bd_inference):
    with gr.Row():
        with gr.Column():
            image_style_input = gr.Image(type="numpy", label="Input Style Image")
            image_content_input = gr.Image(type="numpy", label="Input Content Image")
            prompt = gr.Textbox(label="Prompt",value="best quality, extremely detailed")
            subject_text = gr.Dropdown(
                    ["None", "style", "content", "style & content"], label="blip qformer文本输入", value="style"
                )
        with gr.Column():
            image_output = gr.Gallery(label="Result", ).style(columns=[2], rows=[2], object_fit="contain", height='auto')
            image_button = gr.Button("Generate")
    with gr.Row():
        batch_size = gr.Slider(1, 8, value=4, step=1, label="出图数量(batch_size)")
        ddim = gr.Radio(
                    choices=["ddim", "Euler a"],
                    label=f"Sampler",
                    interactive=True,
                    value="ddim",
                )
        ddim_steps = gr.Slider(10, 50, value=50, step=1, label="采样步数(Steps)", info="Choose between 10 and 50")
        scale = gr.Slider(5, 15, value=8, step=1, label="描述词关联度(CFG scale)", info="Choose between 5 and 15")
        img_weight = gr.Slider(0, 2, value=1.0, step=0.1, label="img embedding加权权重", info="Choose between 0 and 1")
        seed = gr.Number(value=-1,minimum=-1,step=1,label="随机种子(Seed)",info="input -1 for random generation")

    inputs=[
            prompt,
            image_style_input,
            image_content_input,
            subject_text,
            batch_size,
            ddim,
            ddim_steps,
            scale,
            img_weight,
            seed
        ]
    
    image_button.click(bd_inference.generate, inputs=inputs, outputs=image_output)


if __name__ == "__main__":
    inference = DEADiff_CannyControl("configs/inference_deadiff_control_512x512.yaml", "pretrained/deadiff_v1.ckpt", "pretrained/control_sd15_canny.pth")
    with gr.Blocks() as demo:
        gr.HTML(
            """
            <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
            <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
                DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations
            </h1>
            </div>
            """)
        gr.Markdown("Create images in any style of a reference image conditioned on the canny map of one content image using this demo.")
        interface(inference)
    demo.queue(max_size=3)
    demo.launch(server_name="0.0.0.0", server_port=8732)
