import math
import numpy as np
import torch
import safetensors.torch as sf
from PIL import Image
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

SD_NAME = "runwayml/stable-diffusion-v1-5"
MODEL_PATH = "/src/models/iclight_sd15_fbc.safetensors"


class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[setup] device={self.device}")

        print("[setup] Loading SD1.5 components...")
        self.tokenizer = CLIPTokenizer.from_pretrained(SD_NAME, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(SD_NAME, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(SD_NAME, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(SD_NAME, subfolder="unet")

        # Patch UNet: 4 → 12 channels (noisy_latent-4 + fg_latent-4 + bg_latent-4)
        print("[setup] Patching UNet conv_in: 4 → 12 channels...")
        with torch.no_grad():
            new_conv = torch.nn.Conv2d(
                12,
                unet.conv_in.out_channels,
                unet.conv_in.kernel_size,
                unet.conv_in.stride,
                unet.conv_in.padding,
            )
            new_conv.weight.zero_()
            new_conv.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            new_conv.bias = unet.conv_in.bias
            unet.conv_in = new_conv

        # Hook UNet forward to inject concat_conds
        _unet_forward = unet.forward

        def hooked_forward(sample, timestep, encoder_hidden_states, **kwargs):
            c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
            c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
            new_sample = torch.cat([sample, c_concat], dim=1)
            kwargs["cross_attention_kwargs"] = {}
            return _unet_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

        unet.forward = hooked_forward

        # Load IC-Light FBC weights as OFFSETS and add to UNet weights
        print("[setup] Loading IC-Light FBC offsets...")
        sd_offset = sf.load_file(MODEL_PATH)
        sd_origin = unet.state_dict()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged

        # Move to device
        print("[setup] Moving models to device...")
        self.text_encoder = self.text_encoder.to(device=self.device, dtype=torch.float16)
        self.vae = self.vae.to(device=self.device, dtype=torch.bfloat16)
        self.unet = unet.to(device=self.device, dtype=torch.float16)

        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae.set_attn_processor(AttnProcessor2_0())

        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1,
        )

        self.pipe = StableDiffusionPipeline(
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

        print("[setup] Done.")

    # ── Helpers (from demo_fbc.py) ──────────────────────────────────────

    @torch.inference_mode()
    def _encode_prompt_inner(self, txt: str):
        max_length = self.tokenizer.model_max_length
        chunk_length = max_length - 2
        id_start = self.tokenizer.bos_token_id
        id_end = self.tokenizer.eos_token_id
        id_pad = id_end

        def pad(x, p, i):
            return x[:i] if len(x) >= i else x + [p] * (i - len(x))

        tokens = self.tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[id_start] + tokens[i:i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
        chunks = [pad(ck, id_pad, max_length) for ck in chunks]
        token_ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        return self.text_encoder(token_ids).last_hidden_state

    @torch.inference_mode()
    def _encode_prompt_pair(self, positive_prompt: str, negative_prompt: str):
        c = self._encode_prompt_inner(positive_prompt)
        uc = self._encode_prompt_inner(negative_prompt)

        c_len, uc_len = float(len(c)), float(len(uc))
        max_count = max(c_len, uc_len)
        c_repeat = int(math.ceil(max_count / c_len))
        uc_repeat = int(math.ceil(max_count / uc_len))
        max_chunk = max(len(c), len(uc))

        c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
        uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]
        c = torch.cat([p[None, ...] for p in c], dim=1)
        uc = torch.cat([p[None, ...] for p in uc], dim=1)
        return c, uc

    @torch.inference_mode()
    def _numpy2pytorch(self, imgs):
        h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
        return h.movedim(-1, 1)

    @torch.inference_mode()
    def _pytorch2numpy(self, imgs):
        results = []
        for x in imgs:
            y = x.movedim(0, -1) * 127.5 + 127.5
            results.append(y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8))
        return results

    @staticmethod
    def _resize_and_center_crop(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        pil = Image.fromarray(image)
        ow, oh = pil.size
        scale = max(target_w / ow, target_h / oh)
        rw, rh = int(round(ow * scale)), int(round(oh * scale))
        pil = pil.resize((rw, rh), Image.LANCZOS)
        left = (rw - target_w) / 2
        top = (rh - target_h) / 2
        return np.array(pil.crop((left, top, left + target_w, top + target_h)))

    @staticmethod
    def _make_gradient_bg(direction: str, H: int, W: int) -> np.ndarray:
        """Generate a gradient background image encoding light direction."""

        def horiz(left_bright: bool) -> np.ndarray:
            g = np.linspace(224, 32, W) if left_bright else np.linspace(32, 224, W)
            return np.stack([np.tile(g, (H, 1))] * 3, axis=-1).astype(np.uint8)

        def vert(top_bright: bool) -> np.ndarray:
            g = (np.linspace(224, 32, H) if top_bright else np.linspace(32, 224, H))[:, None]
            return np.stack([np.tile(g, (1, W))] * 3, axis=-1).astype(np.uint8)

        d = direction.lower()
        mapping = {
            "left":         lambda: horiz(True),
            "right":        lambda: horiz(False),
            "top":          lambda: vert(True),
            "bottom":       lambda: vert(False),
            "top-left":     lambda: np.clip((horiz(True).astype(np.float32) + vert(True).astype(np.float32)) / 2, 0, 255).astype(np.uint8),
            "top-right":    lambda: np.clip((horiz(False).astype(np.float32) + vert(True).astype(np.float32)) / 2, 0, 255).astype(np.uint8),
            "bottom-left":  lambda: np.clip((horiz(True).astype(np.float32) + vert(False).astype(np.float32)) / 2, 0, 255).astype(np.uint8),
            "bottom-right": lambda: np.clip((horiz(False).astype(np.float32) + vert(False).astype(np.float32)) / 2, 0, 255).astype(np.uint8),
        }
        return mapping.get(d, lambda: np.full((H, W, 3), 64, dtype=np.uint8))()

    # ── Predict ─────────────────────────────────────────────────────────

    def predict(
        self,
        image: Path = Input(description="Kaynak resim"),
        light_direction: str = Input(
            description="Işık yönü",
            default="left",
            choices=["left", "right", "top", "bottom", "top-left", "top-right", "bottom-left", "bottom-right"],
        ),
        prompt: str = Input(description="Ek prompt", default=""),
        steps: int = Input(description="Diffusion adım sayısı", default=25, ge=10, le=50),
        seed: int = Input(description="Seed", default=12345),
    ) -> Path:
        W, H = 512, 512

        # Load and resize input image
        print(f"[predict] Loading image: {image}")
        img_np = np.array(Image.open(str(image)).convert("RGB"))
        fg = self._resize_and_center_crop(img_np, W, H)

        # Build gradient background from light direction
        bg = self._make_gradient_bg(light_direction, H, W)
        print(f"[predict] light_direction={light_direction}")

        # Encode fg + bg as concat conditioning latents
        concat_conds = self._numpy2pytorch([fg, bg]).to(device=self.vae.device, dtype=self.vae.dtype)
        concat_conds = self.vae.encode(concat_conds).latent_dist.mode() * self.vae.config.scaling_factor
        concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)  # 1, 8, h, w

        # Text embeddings
        a_prompt = "best quality"
        n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
        pos = f"{prompt}, {a_prompt}".strip(", ") if prompt else a_prompt
        conds, unconds = self._encode_prompt_pair(pos, n_prompt)

        # Run diffusion
        rng = torch.Generator(device=self.device).manual_seed(seed)
        print(f"[predict] Running pipeline ({steps} steps, seed={seed})...")
        latents = self.pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=W,
            height=H,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            generator=rng,
            output_type="latent",
            guidance_scale=7.0,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(self.vae.dtype) / self.vae.config.scaling_factor

        # Decode
        pixels = self.vae.decode(latents).sample
        result_np = self._pytorch2numpy(pixels)[0]

        out = "/tmp/output.png"
        Image.fromarray(result_np).save(out)
        print("[predict] Done.")
        return Path(out)
