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
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[setup] device={self.device}")

        print("[setup] Loading SD1.5 base...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        # Patch UNet: 4ch → 8ch  (noisy_latents + fg_cond_latents)
        print("[setup] Patching UNet for 8-channel input...")
        orig = self.pipe.unet.conv_in
        new_conv = torch.nn.Conv2d(8, orig.out_channels, orig.kernel_size, orig.stride, orig.padding).to(dtype=self.dtype)
        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, :orig.in_channels] = orig.weight.to(self.dtype)
            new_conv.bias.copy_(orig.bias.to(self.dtype))
        self.pipe.unet.conv_in = new_conv
        self.pipe.unet.config["in_channels"] = 8

        print("[setup] Loading IC-Light weights...")
        state = load_file("/src/models/iclight_sd15_fc.safetensors")
        missing, unexpected = self.pipe.unet.load_state_dict(state, strict=False)
        print(f"[setup] missing={len(missing)} unexpected={len(unexpected)}")
        self.pipe.unet = self.pipe.unet.to(self.device, dtype=self.dtype)

        print("[setup] Loading rembg...")
        self.rembg_session = rembg_session("u2net")
        print("[setup] Done.")

    # ── helpers ──────────────────────────────────────────────────────────

    def _download(self, url: str) -> Image.Image:
        r = subprocess.run(["curl", "-s", "-L", "-o", "/tmp/input.jpg", url], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"Download failed: {r.stderr.decode()}")
        return Image.open("/tmp/input.jpg").convert("RGB")

    def _mask(self, img: Image.Image) -> np.ndarray:
        rgba = rembg_remove(img, session=self.rembg_session)
        alpha = np.array(rgba.split()[-1]).astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        return np.clip(alpha, 0.0, 1.0)

    @staticmethod
    def _dir_text(deg: float, h: float) -> str:
        dirs = ["from the right", "from the upper right", "from above",
                "from the upper left", "from the left", "from the lower left",
                "from below", "from the lower right"]
        idx = int((deg + 22.5) % 360 / 45)
        t = dirs[idx]
        if h > 0.75:
            t = "from directly overhead"
        elif h < 0.15:
            t = t.replace("from", "low").replace("from above", "from the horizon")
        return t

    def _encode_fg(self, img: Image.Image, mask: np.ndarray) -> torch.Tensor:
        """Encode foreground-on-gray as conditioning latent (deterministic .mode())."""
        img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        m3 = np.stack([mask] * 3, axis=-1)
        fg_np = img_np * m3 + 0.5 * (1.0 - m3)  # gray background
        t = torch.from_numpy(fg_np).permute(2, 0, 1).unsqueeze(0)
        t = (t * 2.0 - 1.0).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            # .mode() is DETERMINISTIC — no randomness → no noise
            lat = self.pipe.vae.encode(t).latent_dist.mode()
            lat = lat * self.pipe.vae.config.scaling_factor
        return lat

    def _text_emb(self, prompt: str):
        def enc(txt):
            ids = self.pipe.tokenizer(
                txt, padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids
            with torch.no_grad():
                return self.pipe.text_encoder(ids.to(self.device))[0]
        return torch.cat([enc(""), enc(prompt)])

    def _relight(self, img: Image.Image, mask: np.ndarray, prompt: str, steps: int) -> Image.Image:
        W, H = img.size

        cond = self._encode_fg(img, mask)               # 1,4,h,w  deterministic
        cond2 = cond.repeat(2, 1, 1, 1)                 # 2,4,h,w  for CFG
        emb = self._text_emb(prompt)                    # 2,seq,dim

        # Pure noise start
        latents = torch.randn((1, 4, H // 8, W // 8), device=self.device, dtype=self.dtype)
        self.pipe.scheduler.set_timesteps(steps)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        # IC-Light FC uses low guidance (2.0) — high values cause hallucination
        guidance_scale = 2.0

        for t in self.pipe.scheduler.timesteps:
            lat2 = latents.repeat(2, 1, 1, 1)
            model_in = torch.cat([lat2, cond2], dim=1)  # 2,8,h,w
            model_in = self.pipe.scheduler.scale_model_input(model_in, t)

            with torch.no_grad():
                noise_pred = self.pipe.unet(model_in, t, encoder_hidden_states=emb).sample

            uncond, cond_pred = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (cond_pred - uncond)
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

        with torch.no_grad():
            decoded = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
        decoded = (decoded.clamp(-1, 1) + 1.0) / 2.0
        arr = decoded.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return Image.fromarray((arr * 255).astype(np.uint8))

    def _composite(self, orig: Image.Image, relit: Image.Image, mask: np.ndarray, W: int, H: int) -> Image.Image:
        relit_r = relit.resize((W, H), Image.LANCZOS)
        mf = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
        mf = np.clip(mf, 0.0, 1.0)
        o = np.array(orig).astype(np.float32)
        r = np.array(relit_r).astype(np.float32)
        m = np.stack([mf] * 3, axis=-1)
        result = o * (1.0 - m) + r * m
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    # ── predict ──────────────────────────────────────────────────────────

    def predict(
        self,
        image_url: str = Input(description="Kaynak resim URL"),
        light_direction: float = Input(description="Işık yönü 0-360", default=45.0, ge=0.0, le=360.0),
        light_height: float = Input(description="Işık yüksekliği 0-1", default=0.5, ge=0.0, le=1.0),
        prompt: str = Input(description="Ek prompt", default=""),
    ) -> Path:
        print(f"[predict] {image_url}")
        img = self._download(image_url)
        W, H = img.size

        target = 512
        img512 = img.resize((target, target), Image.LANCZOS).convert("RGB")

        print("[predict] Extracting mask...")
        mask = self._mask(img512)

        dir_text = self._dir_text(light_direction, light_height)
        full_prompt = f"cinematic lighting {dir_text}, soft natural shadows, photorealistic"
        if prompt:
            full_prompt += f", {prompt}"
        print(f"[predict] prompt: {full_prompt}")

        print("[predict] Running IC-Light...")
        relit = self._relight(img512, mask, full_prompt, steps=30)

        print("[predict] Compositing...")
        result = self._composite(img, relit, mask, W, H)

        out = "/tmp/relit_output.png"
        result.save(out, format="PNG")
        print("[predict] Done.")
        return Path(out)
