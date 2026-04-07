"""
IC-Light FC (Foreground Conditioned) — Cog predictor
Reference: https://github.com/lllyasviel/IC-Light/blob/main/gradio_demo.py
"""

import math
import os
import random
import numpy as np
import torch
import safetensors.torch as sf
from PIL import Image
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

SD_NAME   = "runwayml/stable-diffusion-v1-5"
MODEL_PATH = "/src/models/iclight_sd15_fc.safetensors"
MODEL_URL  = "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors"

# ── Tuning defaults (mirror gradio_demo.py) ──────────────────────────────────
A_PROMPT = "best quality"
N_PROMPT = "lowres, bad anatomy, bad hands, cropped, worst quality"
CFG             = 2.0
LOWRES_DENOISE  = 0.9   # strength for initial latent i2i pass
HIGHRES_SCALE   = 1.5   # upscale factor
HIGHRES_DENOISE = 0.5   # strength for highres i2i pass
IMAGE_W, IMAGE_H = 512, 640


class Predictor(BasePredictor):

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        # Disable flash / memory-efficient attention — incompatible with CUDA 11.8
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[setup] device={self.device}")

        # Download FC weights if missing
        if not os.path.exists(MODEL_PATH):
            print("[setup] Downloading iclight_sd15_fc.safetensors …")
            os.makedirs("/src/models", exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[setup] Download complete.")

        # ── Load SD 1.5 components ───────────────────────────────────────────
        print("[setup] Loading SD1.5 components …")
        self.tokenizer    = CLIPTokenizer.from_pretrained(SD_NAME, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(SD_NAME, subfolder="text_encoder")
        self.vae          = AutoencoderKL.from_pretrained(SD_NAME, subfolder="vae")
        unet              = UNet2DConditionModel.from_pretrained(SD_NAME, subfolder="unet")

        # ── Patch UNet conv_in: 4 → 8 channels (FC model) ───────────────────
        # Channel layout: [noisy_latent(4) | fg_latent(4)]
        print("[setup] Patching UNet conv_in: 4 → 8 channels …")
        with torch.no_grad():
            new_conv = torch.nn.Conv2d(
                8,
                unet.conv_in.out_channels,
                unet.conv_in.kernel_size,
                unet.conv_in.stride,
                unet.conv_in.padding,
            )
            new_conv.weight.zero_()
            new_conv.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            new_conv.bias = unet.conv_in.bias
            unet.conv_in = new_conv

        # ── Hook forward to inject fg concat_conds ───────────────────────────
        unet_original_forward = unet.forward

        def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
            c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
            c_concat = torch.cat(
                [c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0
            )
            new_sample = torch.cat([sample, c_concat], dim=1)
            kwargs["cross_attention_kwargs"] = {}
            return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

        unet.forward = hooked_unet_forward

        # ── Merge FC offset weights ──────────────────────────────────────────
        print("[setup] Loading & merging IC-Light FC offset weights …")
        sd_offset = sf.load_file(MODEL_PATH)
        sd_origin = unet.state_dict()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged

        # ── Move to device ───────────────────────────────────────────────────
        self.text_encoder = self.text_encoder.to(device=self.device, dtype=torch.float16)
        self.vae          = self.vae.to(device=self.device, dtype=torch.bfloat16)
        self.unet         = unet.to(device=self.device, dtype=torch.float16)

        # ── Build schedulers & pipelines ────────────────────────────────────
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        shared = dict(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None,
        )
        self.t2i_pipe = StableDiffusionPipeline(**shared)
        self.i2i_pipe = StableDiffusionImg2ImgPipeline(**shared)

        print("[setup] Done.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _encode_prompt_inner(self, txt: str):
        max_length   = self.tokenizer.model_max_length
        chunk_length = max_length - 2
        id_start = self.tokenizer.bos_token_id
        id_end   = self.tokenizer.eos_token_id

        def pad(x, p, i):
            return x[:i] if len(x) >= i else x + [p] * (i - len(x))

        tokens = self.tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [
            [id_start] + tokens[i: i + chunk_length] + [id_end]
            for i in range(0, max(len(tokens), 1), chunk_length)
        ]
        chunks = [pad(ck, id_end, max_length) for ck in chunks]

        token_ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        return self.text_encoder(token_ids).last_hidden_state

    @torch.inference_mode()
    def _encode_prompt_pair(self, positive: str, negative: str):
        c  = self._encode_prompt_inner(positive)
        uc = self._encode_prompt_inner(negative)

        c_len, uc_len = float(len(c)), float(len(uc))
        max_count     = max(c_len, uc_len)
        c  = torch.cat([c]  * int(math.ceil(max_count / c_len)),  dim=0)[:int(max(len(c), len(uc)))]
        uc = torch.cat([uc] * int(math.ceil(max_count / uc_len)), dim=0)[:int(max(len(c), len(uc)))]

        c  = torch.cat([p[None] for p in c],  dim=1)
        uc = torch.cat([p[None] for p in uc], dim=1)
        return c, uc

    @staticmethod
    def _numpy2pytorch(imgs):
        h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
        return h.movedim(-1, 1)

    @staticmethod
    def _pytorch2numpy(imgs):
        results = []
        for x in imgs:
            y = x.movedim(0, -1)
            y = y * 127.5 + 127.5
            results.append(y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8))
        return results

    @staticmethod
    def _resize_and_center_crop(image: np.ndarray, tw: int, th: int) -> np.ndarray:
        pil = Image.fromarray(image)
        ow, oh = pil.size
        scale  = max(tw / ow, th / oh)
        rw, rh = int(round(ow * scale)), int(round(oh * scale))
        pil    = pil.resize((rw, rh), Image.LANCZOS)
        l = (rw - tw) // 2
        t = (rh - th) // 2
        return np.array(pil.crop((l, t, l + tw, t + th)))

    @staticmethod
    def _resize_without_crop(image: np.ndarray, tw: int, th: int) -> np.ndarray:
        return np.array(Image.fromarray(image).resize((tw, th), Image.LANCZOS))

    @staticmethod
    def _angle_to_bg_source(deg: float) -> str:
        """
        Convert light direction angle to IC-Light bg_source.
        0°=right, 90°=top, 180°=left, 270°=bottom
        """
        deg = deg % 360
        if deg < 45 or deg >= 315:
            return "right"
        elif deg < 135:
            return "top"
        elif deg < 225:
            return "left"
        else:
            return "bottom"

    @staticmethod
    def _make_gradient_bg(source: str, w: int, h: int) -> np.ndarray:
        if source == "left":
            g = np.linspace(255, 0, w)
            img = np.tile(g, (h, 1))
        elif source == "right":
            g = np.linspace(0, 255, w)
            img = np.tile(g, (h, 1))
        elif source == "top":
            g = np.linspace(255, 0, h)[:, None]
            img = np.tile(g, (1, w))
        else:  # bottom
            g = np.linspace(0, 255, h)[:, None]
            img = np.tile(g, (1, w))
        return np.stack([img] * 3, axis=-1).astype(np.uint8)

    # ── Core inference ────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _run(
        self,
        fg_rgb: np.ndarray,      # H×W×3 uint8 fg composited on gray
        prompt: str,
        steps: int,
        seed: int,
        light_direction: float,
    ) -> np.ndarray:

        rng = torch.Generator(device=self.device).manual_seed(seed)

        # ── Prompts ──────────────────────────────────────────────────────────
        full_positive = f"{prompt}, {A_PROMPT}" if prompt else A_PROMPT
        conds, unconds = self._encode_prompt_pair(full_positive, N_PROMPT)

        # ── Resize fg & encode for concat_cond ───────────────────────────────
        fg = self._resize_and_center_crop(fg_rgb, IMAGE_W, IMAGE_H)
        fg_t = self._numpy2pytorch([fg]).to(device=self.vae.device, dtype=self.vae.dtype)
        concat_conds = (
            self.vae.encode(fg_t).latent_dist.mode() * self.vae.config.scaling_factor
        )

        # ── Build initial latent from gradient background ─────────────────────
        bg_source = self._angle_to_bg_source(light_direction)
        bg_np  = self._make_gradient_bg(bg_source, IMAGE_W, IMAGE_H)
        bg     = self._resize_and_center_crop(bg_np, IMAGE_W, IMAGE_H)
        bg_t   = self._numpy2pytorch([bg]).to(device=self.vae.device, dtype=self.vae.dtype)
        bg_latent = (
            self.vae.encode(bg_t).latent_dist.mode() * self.vae.config.scaling_factor
        )

        print(f"[run] bg_source={bg_source} steps={steps} seed={seed}")

        # ── Low-res i2i pass (bg → guided by fg concat) ──────────────────────
        latents = self.i2i_pipe(
            image=bg_latent,
            strength=LOWRES_DENOISE,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=IMAGE_W,
            height=IMAGE_H,
            num_inference_steps=int(round(steps / LOWRES_DENOISE)),
            num_images_per_prompt=1,
            generator=rng,
            output_type="latent",
            guidance_scale=CFG,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(self.vae.dtype) / self.vae.config.scaling_factor

        # ── Decode & upscale ─────────────────────────────────────────────────
        pixels = self.vae.decode(latents).sample
        pixels = self._pytorch2numpy(pixels)

        hr_w = int(round(IMAGE_W * HIGHRES_SCALE / 64.0) * 64)
        hr_h = int(round(IMAGE_H * HIGHRES_SCALE / 64.0) * 64)
        pixels = [self._resize_without_crop(p, hr_w, hr_h) for p in pixels]

        pixels_t = self._numpy2pytorch(pixels).to(device=self.vae.device, dtype=self.vae.dtype)
        latents   = self.vae.encode(pixels_t).latent_dist.mode() * self.vae.config.scaling_factor
        latents   = latents.to(device=self.unet.device, dtype=self.unet.dtype)

        # Re-encode fg at highres size
        fg_hr  = self._resize_and_center_crop(fg_rgb, hr_w, hr_h)
        fg_hr_t = self._numpy2pytorch([fg_hr]).to(device=self.vae.device, dtype=self.vae.dtype)
        concat_conds = (
            self.vae.encode(fg_hr_t).latent_dist.mode() * self.vae.config.scaling_factor
        )

        # ── High-res i2i pass ────────────────────────────────────────────────
        latents = self.i2i_pipe(
            image=latents,
            strength=HIGHRES_DENOISE,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=hr_w,
            height=hr_h,
            num_inference_steps=int(round(steps / HIGHRES_DENOISE)),
            num_images_per_prompt=1,
            generator=rng,
            output_type="latent",
            guidance_scale=CFG,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(self.vae.dtype) / self.vae.config.scaling_factor

        pixels = self.vae.decode(latents).sample
        result = self._pytorch2numpy(pixels)[0]   # H×W×3 uint8
        return result

    # ── Cog predict ──────────────────────────────────────────────────────────

    def predict(
        self,
        image: Path = Input(description="Input image (person or subject)"),
        prompt: str = Input(
            description="Lighting description",
            default="soft studio lighting",
        ),
        light_direction: float = Input(
            description="Light direction in degrees: 0=right, 90=top, 180=left, 270=bottom",
            default=0.0,
            ge=0.0,
            le=360.0,
        ),
        steps: int = Input(
            description="Number of diffusion steps",
            default=25,
            ge=10,
            le=50,
        ),
        seed: int = Input(
            description="Random seed (-1 = random)",
            default=-1,
        ),
    ) -> Path:

        if seed == -1:
            seed = random.randint(0, 2**31)
        print(f"[predict] prompt={prompt!r} dir={light_direction} steps={steps} seed={seed}")

        # ── Load & foreground-extract with rembg ─────────────────────────────
        from rembg import remove as rembg_remove

        src_pil = Image.open(str(image)).convert("RGB")
        src_np  = np.array(src_pil)

        print("[predict] Running rembg …")
        fg_rgba = rembg_remove(src_pil)                  # PIL RGBA
        fg_np   = np.array(fg_rgba.convert("RGBA"))

        # Composite fg onto gray (127) background — same as official run_rmbg(sigma=0)
        alpha   = fg_np[:, :, 3:4].astype(np.float32) / 255.0
        rgb     = fg_np[:, :, :3].astype(np.float32)
        gray    = np.full_like(rgb, 127.0)
        fg_rgb  = (rgb * alpha + gray * (1.0 - alpha)).clip(0, 255).astype(np.uint8)

        # ── Run IC-Light FC ───────────────────────────────────────────────────
        result_np = self._run(fg_rgb, prompt, steps, seed, light_direction)

        # ── Save output ───────────────────────────────────────────────────────
        out_path = "/tmp/output.png"
        Image.fromarray(result_np).save(out_path)
        print(f"[predict] Saved → {out_path}")
        return Path(out_path)
