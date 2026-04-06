import subprocess
import numpy as np
import torch
import cv2
from PIL import Image
from cog import BasePredictor, Input, Path

from diffusers import StableDiffusionPipeline, DDIMScheduler
from safetensors.torch import load_file
from rembg import remove as rembg_remove, new_session as rembg_session


class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[setup] device={self.device}, dtype={self.model_dtype}")

        # ── Load IC-Light pipeline ──────────────────────────────────────
        print("[setup] Loading Stable Diffusion 1.5 base...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=self.model_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        # Patch UNet to accept 8-channel input
        # IC-Light fc: [noisy_latents(4) + fg_cond_latents(4)] = 8 channels
        print("[setup] Patching UNet for 8-channel input...")
        orig_conv = self.pipe.unet.conv_in
        new_conv = torch.nn.Conv2d(
            8,
            orig_conv.out_channels,
            orig_conv.kernel_size,
            orig_conv.stride,
            orig_conv.padding,
        ).to(dtype=self.model_dtype)
        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, :orig_conv.in_channels] = orig_conv.weight.to(dtype=self.model_dtype)
            new_conv.bias.copy_(orig_conv.bias.to(dtype=self.model_dtype))
        self.pipe.unet.conv_in = new_conv
        self.pipe.unet.config["in_channels"] = 8

        print("[setup] Loading IC-Light UNet weights...")
        ic_light_state = load_file("/src/models/iclight_sd15_fc.safetensors")
        missing, unexpected = self.pipe.unet.load_state_dict(ic_light_state, strict=False)
        print(f"[setup] IC-Light loaded — missing={len(missing)}, unexpected={len(unexpected)}")
        self.pipe.unet = self.pipe.unet.to(self.device, dtype=self.model_dtype)

        # ── Load rembg session ──────────────────────────────────────────
        print("[setup] Loading rembg (u2net) session...")
        self.rembg_session = rembg_session("u2net")
        print("[setup] Done.")

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _direction_to_text(direction_deg: float, height: float) -> str:
        """Convert numeric direction/height to a natural language lighting description."""
        dirs = [
            "from the right",
            "from the upper right",
            "from above",
            "from the upper left",
            "from the left",
            "from the lower left",
            "from below",
            "from the lower right",
        ]
        idx = int((direction_deg + 22.5) % 360 / 45)
        dir_text = dirs[idx]
        if height > 0.75:
            dir_text = "from directly overhead"
        elif height < 0.15:
            dir_text = dir_text.replace("from", "from low").replace("from above", "from the horizon")
        return dir_text

    def _download_image(self, url: str) -> Image.Image:
        result = subprocess.run(
            ["curl", "-s", "-L", "-o", "/tmp/input_image.jpg", url],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Image download failed: {result.stderr.decode()}")
        return Image.open("/tmp/input_image.jpg").convert("RGB")

    def _extract_mask(self, img: Image.Image) -> np.ndarray:
        """Returns float32 H×W mask in [0,1] using rembg."""
        result_rgba = rembg_remove(img, session=self.rembg_session)
        alpha = np.array(result_rgba.split()[-1]).astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        return np.clip(alpha, 0.0, 1.0)

    def _encode_fg_conditioning(self, img: Image.Image, mask: np.ndarray) -> torch.Tensor:
        """
        Build the foreground conditioning latent for IC-Light FC.
        Subject is kept, background is replaced with neutral gray (0.5).
        """
        img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        gray_bg = np.full_like(img_np, 0.5)
        fg_np = img_np * mask_3ch + gray_bg * (1.0 - mask_3ch)

        fg_tensor = torch.from_numpy(fg_np).permute(2, 0, 1).unsqueeze(0)
        fg_tensor = (fg_tensor * 2.0 - 1.0).to(self.device, dtype=self.model_dtype)

        with torch.no_grad():
            latent = self.pipe.vae.encode(fg_tensor).latent_dist.sample()
            latent = latent * self.pipe.vae.config.scaling_factor
        return latent

    def _build_text_embeddings(self, prompt: str):
        def encode(text):
            inputs = self.pipe.tokenizer(
                text,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                return self.pipe.text_encoder(inputs.input_ids.to(self.device))[0]

        text_emb = encode(prompt)
        uncond_emb = encode("")
        return torch.cat([uncond_emb, text_emb])

    def _run_ic_light(
        self,
        img: Image.Image,
        mask: np.ndarray,
        prompt: str,
        num_steps: int,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """Run the IC-Light FC diffusion pipeline and return the relighted image."""
        W, H = img.size

        # 1. Foreground conditioning latent
        cond_latents = self._encode_fg_conditioning(img, mask)           # 1,4,h,w
        cond_latents_cfg = torch.cat([cond_latents, cond_latents], dim=0)  # 2,4,h,w (for CFG)

        # 2. Text embeddings (uncond + cond)
        encoder_hidden_states = self._build_text_embeddings(prompt)      # 2, seq, dim

        # 3. Start from pure noise
        latent_h, latent_w = H // 8, W // 8
        latents = torch.randn(
            (1, 4, latent_h, latent_w),
            device=self.device,
            dtype=self.model_dtype,
        )

        self.pipe.scheduler.set_timesteps(num_steps)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        # 4. Denoising loop
        for t in self.pipe.scheduler.timesteps:
            latents_cfg = torch.cat([latents, latents], dim=0)  # 2,4,h,w

            # Concatenate noisy latents with foreground conditioning → 8 channels
            model_input = torch.cat([latents_cfg, cond_latents_cfg], dim=1)  # 2,8,h,w
            model_input = self.pipe.scheduler.scale_model_input(model_input, t)

            with torch.no_grad():
                noise_pred = self.pipe.unet(
                    model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # 5. Decode
        with torch.no_grad():
            decoded = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
        decoded = (decoded.clamp(-1, 1) + 1.0) / 2.0
        decoded_np = decoded.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return Image.fromarray((decoded_np * 255).astype(np.uint8))

    def _composite(
        self,
        original: Image.Image,
        relit: Image.Image,
        mask_512: np.ndarray,
        orig_W: int,
        orig_H: int,
    ) -> Image.Image:
        relit_resized = relit.resize((orig_W, orig_H), Image.LANCZOS)
        mask_full = cv2.resize(mask_512, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
        mask_full = np.clip(mask_full, 0.0, 1.0)

        orig_np = np.array(original).astype(np.float32)
        relit_np = np.array(relit_resized).astype(np.float32)
        mask_3ch = np.stack([mask_full, mask_full, mask_full], axis=-1)

        result_np = orig_np * (1.0 - mask_3ch) + relit_np * mask_3ch
        return Image.fromarray(np.clip(result_np, 0, 255).astype(np.uint8))

    # ── Public predict ───────────────────────────────────────────────────

    def predict(
        self,
        image_url: str = Input(description="URL of the input image to relight"),
        light_direction: float = Input(
            description="Light direction in degrees (0=right, 90=up, 180=left, 270=down)",
            default=45.0,
            ge=0.0,
            le=360.0,
        ),
        light_height: float = Input(
            description="Light height (0=horizon, 1=directly overhead)",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        prompt: str = Input(
            description="Optional extra text to add to the lighting prompt",
            default="",
        ),
        num_steps: int = Input(
            description="Number of diffusion steps",
            default=30,
            ge=10,
            le=50,
        ),
    ) -> Path:
        # 1. Download + convert
        print(f"[predict] Downloading image: {image_url}")
        img_pil = self._download_image(image_url)
        W, H = img_pil.size
        print(f"[predict] Image size: {W}x{H}")

        target_size = 512
        img_resized = img_pil.resize((target_size, target_size), Image.LANCZOS).convert("RGB")

        # 2. Extract foreground mask
        print("[predict] Extracting foreground mask...")
        mask = self._extract_mask(img_resized)

        # 3. Build direction-aware prompt
        dir_text = self._direction_to_text(light_direction, light_height)
        base_prompt = f"cinematic lighting {dir_text}, soft natural shadows, photorealistic"
        effective_prompt = f"{base_prompt}, {prompt}".strip(", ") if prompt else base_prompt
        print(f"[predict] Prompt: {effective_prompt}")

        # 4. Run IC-Light
        print(f"[predict] Running IC-Light ({num_steps} steps)...")
        relit_pil = self._run_ic_light(img_resized, mask, effective_prompt, num_steps)

        # 5. Composite onto original background
        print("[predict] Compositing...")
        result_pil = self._composite(img_pil, relit_pil, mask, W, H)

        out_path = "/tmp/relit_output.png"
        result_pil.save(out_path, format="PNG")
        print("[predict] Done.")
        return Path(out_path)
