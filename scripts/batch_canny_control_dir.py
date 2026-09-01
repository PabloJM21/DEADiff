# Copyright (2024) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path

import accelerate
import cv2
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision import transforms as T
from torchvision.utils import make_grid

from ldm.controlnet.annotator.canny import CannyDetector
from ldm.controlnet.annotator.util import HWC3, resize_image
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


apply_canny = CannyDetector()


def load_gt_points_from_txt(img_path, w, h):
    txt_path = os.path.splitext(img_path)[0] + ".txt"

    if not os.path.exists(txt_path):
        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()

    if not line:
        return None

    values = line.split()
    if len(values) < 13:
        return None

    coords = list(map(float, values[-8:]))

    x_tl, y_tl, x_bl, y_bl, x_tr, y_tr, x_br, y_br = coords

    gt_pts = np.array(
        [
            [x_tl * w, y_tl * h],
            [x_tr * w, y_tr * h],
            [x_br * w, y_br * h],
            [x_bl * w, y_bl * h],
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)

    return gt_pts


def load_gt_mask(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    gt_mask = None

    gt_pts = load_gt_points_from_txt(img_path, w, h)
    if gt_pts is not None:
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(gt_mask, [gt_pts], 255)

    if gt_mask is None:
        raise ValueError(f"Could not read GT mask: {img_path}")

    if gt_mask.ndim == 3:
        gt_mask = gt_mask[..., 0]

    return (gt_mask > 0).astype(np.float32)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    print("loading done")
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
        c_crossattn = cond["c_crossattn"]
        c_concat = cond["c_concat"]
        uc_crossattn = uncond["c_crossattn"]
        uc_concat = uncond["c_concat"]
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
        cond_in = {
            "c_crossattn": cond_in,
            "c_concat": [torch.cat([uc_concat[0], c_concat[0]], 0)],
        }
        uncond, cond = self.inner_model(
            x_in, sigma_in, cond=cond_in, img_weight=img_weight
        ).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class DEADiffCannyBatch(object):
    def __init__(self, config, ckpt, ckpt_controlnet):
        config = OmegaConf.load(config)
        config.model.params.control_stage_config.params.ckpt_path = ckpt_controlnet
        self.model = load_model_from_config(config, ckpt)
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    def generate(
        self,
        prompt,
        image_style_input,
        image_content_input,
        image_canny_map_input,
        subject_text,
        batch_size,
        sampler_name,
        ddim_steps,
        scale,
        img_weight,
        seed,
        negative_prompt,
        canny_low_threshold,
        canny_high_threshold,
        control_resolution,
        style_image_size,
        precision,
        image_content_input_path,
        use_gt_mask,
    ):
        accelerator = accelerate.Accelerator()
        device = accelerator.device

        if seed < 0:
            seed = torch.randint(0, 2**31 - 1, (1,)).item()
        seed_everything(seed)

        n_rows = 1 if batch_size < 2 else (2 if batch_size < 5 else 3)
        prompts = batch_size * [prompt]

        precision_scope = autocast if precision == "autocast" else torch.cuda.amp.autocast
        precision_enabled = precision == "autocast"
        with torch.no_grad():
            with precision_scope("cuda", enabled=precision_enabled):
                with self.model.ema_scope():
                    if scale != 1.0:
                        uc_encoder_hidden_states = self.model.get_learned_conditioning(
                            {
                                "target_text": batch_size
                                * [negative_prompt],
                                "subject_text": subject_text,
                            }
                        )
                    else:
                        uc_encoder_hidden_states = None

                    if subject_text == "style & content":
                        subject_text = ["style", "content"]
                    if subject_text == "None":
                        subject_text = None

                    img = resize_image(HWC3(image_content_input), control_resolution)
                    h, w, _ = img.shape

                    if image_canny_map_input is not None:
                        boundary_map = HWC3(image_canny_map_input)
                        # Keep boundary alignment with the resized content image.
                        boundary_map = np.array(
                            Image.fromarray(boundary_map).resize((w, h), resample=Image.NEAREST)
                        )
                        detected_map = HWC3(boundary_map)
                    else:
                        detected_map = apply_canny(img, canny_low_threshold, canny_high_threshold)
                        detected_map = HWC3(detected_map)
                    canny_map = Image.fromarray(detected_map)

                    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                    control = torch.stack([control for _ in range(batch_size)], dim=0)
                    control = rearrange(control, "b h w c -> b c h w").clone()

                    c_encoder_hidden_states = self.model.get_learned_conditioning(
                        {
                            "target_text": prompts,
                            "inp_image": 2
                            * (
                                T.ToTensor()(Image.fromarray(image_style_input).convert("RGB").resize((style_image_size, style_image_size)))
                                - 0.5
                            )
                            .unsqueeze(0)
                            .repeat(batch_size, 1, 1, 1)
                            .to("cuda"),
                            "subject_text": [subject_text] * batch_size,
                        }
                    )
                    uc, c = uc_encoder_hidden_states, c_encoder_hidden_states
                    cond = {"c_concat": [control], "c_crossattn": c}
                    un_cond = {"c_concat": [control], "c_crossattn": [uc, uc]}

                    shape = [4, h // 8, w // 8]

                    if sampler_name == "ddim":
                        sampler = DDIMSampler(self.model)
                        samples_ddim, _ = sampler.sample(
                            S=ddim_steps,
                            conditioning=cond,
                            batch_size=batch_size,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=un_cond,
                            img_weight=img_weight,
                        )
                    else:
                        sigmas = self.model_wrap.get_sigmas(ddim_steps)
                        x = torch.randn([batch_size, *shape], device=device) * sigmas[0]
                        extra_args = {
                            "cond": cond,
                            "uncond": un_cond,
                            "cond_scale": scale,
                            "img_weight": img_weight,
                        }
                        samples_ddim = K.sampling.sample_euler_ancestral(
                            self.model_wrap_cfg,
                            x,
                            sigmas,
                            extra_args=extra_args,
                            disable=not accelerator.is_main_process,
                        )

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = accelerator.gather(x_samples_ddim)

                    if use_gt_mask:
                        gt_mask = load_gt_mask(image_content_input_path)
                        target_width = x_samples_ddim.shape[-1]
                        target_height = x_samples_ddim.shape[-2]
                        if gt_mask.shape != (target_height, target_width):
                            gt_mask = cv2.resize(
                                gt_mask,
                                (target_width, target_height),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        gt_mask = torch.from_numpy(gt_mask).float().to(x_samples_ddim.device)
                        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
                        gt_mask = gt_mask.repeat(x_samples_ddim.shape[0], 1, 1, 1)
                        original = torch.from_numpy(img).float().to(x_samples_ddim.device) / 255.0
                        original = rearrange(original, "h w c -> 1 c h w")
                        original = original.repeat(x_samples_ddim.shape[0], 1, 1, 1)
                        x_samples_ddim = x_samples_ddim * (1.0 - gt_mask) + original * gt_mask

                    if accelerator.is_main_process:
                        all_samples = [
                            T.ToPILImage()(x_sample_ddim) for x_sample_ddim in x_samples_ddim
                        ]
                        grid = make_grid(x_samples_ddim, nrow=n_rows)
                        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                        grid_image = Image.fromarray(grid.astype(np.uint8))
                        return all_samples, grid_image, canny_map, seed

                    return [], None, canny_map, seed


def parse_args():
    parser = argparse.ArgumentParser(description="Batch CLI for DEADiff Canny Control (directory mode)")
    parser.add_argument("--style_image", type=str, required=True, help="Path to style reference image")
    parser.add_argument("-content_dir", "--content_dir", type=str, required=True, help="Path to directory of content images")
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        help="File extension to match in content_dir (for example .jpeg)",
    )
    parser.add_argument("--prompt", type=str, default="best quality, extremely detailed")
    parser.add_argument(
        "--canny_map",
        type=str,
        default=None,
        help="Optional path to a precomputed boundary/canny map to use as control",
    )
    parser.add_argument(
        "--subject_text",
        type=str,
        default="style",
        choices=["None", "style", "content", "style & content"],
        help="BLIP subject text mode",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "Euler a"])
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=8.0)
    parser.add_argument("--img_weight", type=float, default=1.0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate",
        help="Negative prompt used for unconditional guidance",
    )
    parser.add_argument("--canny_low_threshold", type=int, default=100)
    parser.add_argument("--canny_high_threshold", type=int, default=200)
    parser.add_argument("--control_resolution", type=int, default=384)
    parser.add_argument("--style_image_size", type=int, default=224)
    parser.add_argument(
        "--precision",
        type=str,
        default="autocast",
        choices=["autocast", "full"],
        help="Inference precision mode",
    )
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--config", type=str, default="configs/inference_deadiff_control_512x512.yaml")
    parser.add_argument("--ckpt", type=str, default="pretrained/deadiff_v1.ckpt")
    parser.add_argument("--control_ckpt", type=str, default="pretrained/control_sd15_canny.pth")
    parser.add_argument("--output_dir", type=str, default="outputs/canny_batch")
    parser.add_argument(
        "--use_gt_mask",
        action="store_true",
        help="Enable runway preservation using the sibling .txt polygon mask for each content image.",
    )
    parser.add_argument(
        "--save_canny_map",
        action="store_true",
        help="Kept for compatibility; output mode only writes final renamed grid images",
    )
    return parser.parse_args()


def normalize_extension(ext):
    ext = ext.strip().lower()
    if not ext:
        raise ValueError("--format must be a non-empty extension")
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def main():
    args = parse_args()

    content_dir = Path(args.content_dir)
    if not content_dir.is_dir():
        raise ValueError(f"content_dir does not exist or is not a directory: {content_dir}")

    target_ext = normalize_extension(args.format)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_files = sorted(
        [p for p in content_dir.iterdir() if p.is_file() and p.suffix.lower() == target_ext]
    )
    if not matching_files:
        raise ValueError(f"No files with extension '{target_ext}' found in {content_dir}")

    style_image = np.array(Image.open(args.style_image).convert("RGB"))
    canny_map_image = None
    if args.canny_map:
        canny_map_image = np.array(Image.open(args.canny_map).convert("RGB"))

    pipeline = DEADiffCannyBatch(args.config, args.ckpt, args.control_ckpt)

    for content_path in matching_files:
        content_image = np.array(Image.open(content_path).convert("RGB"))
        _, grid_image, _, _ = pipeline.generate(
            prompt=args.prompt,
            image_style_input=style_image,
            image_content_input=content_image,
            image_content_input_path=str(content_path),
            image_canny_map_input=canny_map_image,
            subject_text=args.subject_text,
            batch_size=args.batch_size,
            sampler_name=args.sampler,
            ddim_steps=args.ddim_steps,
            scale=args.scale,
            img_weight=args.img_weight,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
            canny_low_threshold=args.canny_low_threshold,
            canny_high_threshold=args.canny_high_threshold,
            control_resolution=args.control_resolution,
            style_image_size=args.style_image_size,
            precision=args.precision,
            use_gt_mask=args.use_gt_mask,
        )

        if grid_image is None:
            continue

        output_path = output_dir / f"{content_path.stem}.jpeg"
        grid_image.convert("RGB").save(output_path, format="JPEG")
        print(f"Saved: {output_path}")

    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()