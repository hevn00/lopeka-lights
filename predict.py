import io
import subprocess
import numpy as np
import torch
import cv2
from PIL import Image
from cog import BasePredictor, Input, Path

from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from safetensors.torch import load_file
from rembg import remove as rembg_remove, new_session as rembg_session


class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[setup] device={self.device}")

        # ── Load IC-Light pipeline ──────────────────────────────────────
        print("[setup] Loading Stable Diffusion 1.5 base...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        # Patch UNet to accept 8-channel input (image RGBA + light map RGB + mask)
        # IC-Light fc variant expects 8 input channels
        print("[setup] Patching UNet for 8-channel input...")
        orig_conv = self.pipe.unet.conv_in
        new_in_channels = 8
        new_conv = torch.nn.Conv2d(
            new_in_channels,
            orig_conv.out_channels,
            orig_conv.kernel_size,
            orig_conv.stride,
            orig_conv.padding,
        )
        # Zero-init extra channels, copy existing weights
        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, :orig_conv.in_channels] = orig_conv.weight
            new_conv.bias.copy_(orig_conv.bias)
        self.pipe.unet.conv_in = new_conv
        self.pipe.unet.config["in_channels"] = new_in_channels

        print("[setup] Loading IC-Light UNet weights...")
        ic_light_state = load_file("/src/models/iclight_sd15_fc.safetensors")
        missing, unexpected = self.pipe.unet.load_state_dict(ic_light_state, strict=False)
        print(f"[setup] IC-Light loaded — missing={len(missing)}, unexpected={len(unexpected)}")
        self.pipe.unet = self.pipe.unet.to(self.device)

        # ── Load rembg segmentation session ────────────────────────────
        print("[setup] Loading rembg (u2net) session...")
        self.rembg_session = rembg_session("u2net")
        print("[setup] Done.")

    # ── Public predict ──────────────────────────────────────────────────

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
            description="Optional text prompt to guide relighting style",
            default="",
        ),
        num_steps: int = Input(
            description="Number of diffusion steps",
            default=25,
            ge=10,
            le=50,
        ),
    ) -> Path:
        # 1. Download input image
        print(f"[predict] Downloading image: {image_url}")
        img_pil = self._download_image(image_url)
        W, H = img_pil.size
        print(f"[predict] Image size: {W}x{H}")

        # Resize to 512x512 for SD pipeline, keep original for final compositing
        target_size = 512
        img_resized = img_pil.resize((target_size, target_size), Image.LANCZOS)

        # 2. Extract foreground mask via BiRefNet
        print("[predict] Extracting foreground mask...")
        mask = self._extract_mask(img_resized)  # H x W float32 [0,1]

        # 3. Generate light map from direction + height
        print(f"[predict] Generating light map (dir={light_direction}, height={light_height})...")
        light_map = self._make_light_map(target_size, target_size, light_direction, light_height)
        light_map_pil = Image.fromarray(light_map, mode="RGB")

        # 4. Run IC-Light pipeline
        print(f"[predict] Running IC-Light pipeline ({num_steps} steps)...")
        relit_pil = self._run_ic_light(
            img_resized, mask, light_map_pil, prompt, num_steps
        )

        # 5. Composite relit foreground onto original background
        print("[predict] Compositing result...")
        result_pil = self._composite(img_pil, relit_pil, mask, W, H)

        # 6. Save and return
        out_path = "/tmp/relit_output.png"
        result_pil.save(out_path, format="PNG")
        print("[predict] Done.")
        return Path(out_path)

    # ── Private helpers ─────────────────────────────────────────────────

    def _download_image(self, url: str) -> Image.Image:
        result = subprocess.run(
            ["curl", "-s", "-L", "-o", "/tmp/input_image.jpg", url],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Image download failed: {result.stderr.decode()}")
        img = Image.open("/tmp/input_image.jpg").convert("RGB")
        return img

    def _extract_mask(self, img: Image.Image) -> np.ndarray:
        """Returns float32 H x W mask in [0, 1] using rembg."""
        # rembg returns RGBA image — alpha channel is the foreground mask
        result_rgba = rembg_remove(img, session=self.rembg_session)
        alpha = np.array(result_rgba.split()[-1]).astype(np.float32) / 255.0

        # Soft edges
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        return np.clip(alpha, 0.0, 1.0)

    @staticmethod
    def _make_light_map(W: int, H: int, direction_deg: float, height: float) -> np.ndarray:
        """
        Generate a directional sphere light map.
        direction_deg: 0=right, 90=up, 180=left, 270=down
        height: 0=horizon, 1=overhead
        Returns: uint8 H x W x 3 RGB image
        """
        angle = np.radians(direction_deg)
        lx = np.cos(angle)
        ly = -np.sin(angle)
        lz = height * 2.0 - 1.0  # map [0,1] → [-1,1]

        # Normalize light direction vector
        norm = np.sqrt(lx**2 + ly**2 + lz**2)
        lx /= norm
        ly /= norm
        lz /= norm

        # Create pixel grid in [-1, 1]
        y_coords, x_coords = np.mgrid[0:H, 0:W]
        x_norm = (x_coords / (W - 1)) * 2.0 - 1.0
        y_norm = (y_coords / (H - 1)) * 2.0 - 1.0

        # Sphere: z = sqrt(max(1 - x^2 - y^2, 0))
        r_sq = x_norm**2 + y_norm**2
        sphere_mask = r_sq <= 1.0
        z_norm = np.sqrt(np.maximum(1.0 - r_sq, 0.0))

        # Diffuse (Lambertian) shading
        diffuse = np.maximum(x_norm * lx + y_norm * ly + z_norm * lz, 0.0)

        # Build 3-channel light map: sphere region lit, outside dark
        light_map = np.zeros((H, W, 3), dtype=np.float32)
        ambient = 0.15  # slight ambient so fully-dark regions still have some light
        light_map[:, :, :] = ambient  # fill with ambient
        light_map[sphere_mask, 0] = diffuse[sphere_mask]
        light_map[sphere_mask, 1] = diffuse[sphere_mask]
        light_map[sphere_mask, 2] = diffuse[sphere_mask]

        # Brighten highlights
        light_map = np.clip(light_map * 1.4, 0.0, 1.0)

        return (light_map * 255).astype(np.uint8)

    def _run_ic_light(
        self,
        img: Image.Image,
        mask: np.ndarray,
        light_map: Image.Image,
        prompt: str,
        num_steps: int,
    ) -> Image.Image:
        """Run the IC-Light diffusion pipeline and return the relighted image."""
        W, H = img.size

        # Convert image to tensor [0,1] float16
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W

        # Normalize to [-1, 1] for SD
        img_tensor = img_tensor * 2.0 - 1.0

        # Foreground mask tensor  1,1,H,W
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # 1,1,H,W

        # Light map tensor  1,3,H,W  in [-1,1]
        lm_np = np.array(light_map).astype(np.float32) / 255.0
        lm_tensor = torch.from_numpy(lm_np).permute(2, 0, 1).unsqueeze(0)
        lm_tensor = lm_tensor * 2.0 - 1.0

        # Concatenate: [image(3) + mask(1) + light_map(3) + ones(1)] = 8 channels
        # IC-Light fc variant: 4 channels image (RGBA) + 4 channels concat (light map RGB + 1)
        ones_tensor = torch.ones_like(mask_tensor)
        concat_input = torch.cat([img_tensor, mask_tensor, lm_tensor, ones_tensor], dim=1)  # 1,8,H,W
        concat_input = concat_input.to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)

        # Encode concatenated input to latent space
        # We use the VAE to encode a 3-channel blend (image * mask + light_map * (1-mask))
        # Then inject the full 8ch as cross-frame conditioning via UNet
        with torch.no_grad():
            # Prepare a visual hint for VAE encoding: overlay light map on subject
            fg_mask_3 = mask_tensor.expand(-1, 3, -1, -1)
            visual_hint = img_tensor * fg_mask_3 + lm_tensor * (1.0 - fg_mask_3)
            visual_hint = visual_hint.to(
                self.device,
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            latents_init = self.pipe.vae.encode(visual_hint).latent_dist.sample()
            latents_init = latents_init * self.pipe.vae.config.scaling_factor

        # Build text embeddings
        effective_prompt = prompt if prompt else "a person with natural directional lighting"
        text_inputs = self.pipe.tokenizer(
            effective_prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]

        uncond_inputs = self.pipe.tokenizer(
            "",
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond_embeddings = self.pipe.text_encoder(
                uncond_inputs.input_ids.to(self.device)
            )[0]

        encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

        # DDIM scheduling
        self.pipe.scheduler.set_timesteps(num_steps)
        timesteps = self.pipe.scheduler.timesteps

        # Add noise to latent init
        noise = torch.randn_like(latents_init)
        latents = self.pipe.scheduler.add_noise(
            latents_init, noise, timesteps[:1].expand(latents_init.shape[0])
        )

        # Prepare IC-Light conditioning: tile concat_input to match latent spatial dims
        latent_h = latents.shape[2]
        latent_w = latents.shape[3]
        # Downsample concat_input channels to latent space via average pooling
        concat_cond = torch.nn.functional.interpolate(
            concat_input, size=(latent_h * 8, latent_w * 8), mode="bilinear", align_corners=False
        )
        # Encode each group with VAE
        with torch.no_grad():
            img_latent = self.pipe.vae.encode(
                concat_input[:, :3].to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)
            ).latent_dist.sample() * self.pipe.vae.config.scaling_factor
            lm_latent = self.pipe.vae.encode(
                concat_input[:, 4:7].to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)
            ).latent_dist.sample() * self.pipe.vae.config.scaling_factor

        # IC-Light expects concatenation of: noisy_latents(4) + img_latent(4) → but we use 8ch UNet
        # So we concatenate noisy latents with our image latent encoding
        guidance_scale = 7.5

        for i, t in enumerate(timesteps):
            latents_input = torch.cat([latents, latents], dim=0)  # batch for CFG
            # Concat conditioning (IC-Light fc: noisy_latents cat with reference latents)
            img_latent_cat = torch.cat([img_latent, img_latent], dim=0)
            model_input = torch.cat([latents_input, img_latent_cat], dim=1)  # B,8,h,w

            with torch.no_grad():
                noise_pred = self.pipe.unet(
                    model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        with torch.no_grad():
            decoded = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
        decoded = (decoded.clamp(-1, 1) + 1.0) / 2.0
        decoded_np = decoded.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        relit_img = Image.fromarray((decoded_np * 255).astype(np.uint8))
        return relit_img

    def _composite(
        self,
        original: Image.Image,
        relit: Image.Image,
        mask_512: np.ndarray,
        orig_W: int,
        orig_H: int,
    ) -> Image.Image:
        """
        Composite the relighted foreground back onto the original background
        at original resolution.
        """
        # Upscale relit and mask to original size
        relit_resized = relit.resize((orig_W, orig_H), Image.LANCZOS)
        mask_full = cv2.resize(mask_512, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
        mask_full = np.clip(mask_full, 0.0, 1.0)

        orig_np = np.array(original).astype(np.float32)
        relit_np = np.array(relit_resized).astype(np.float32)
        mask_3ch = np.stack([mask_full, mask_full, mask_full], axis=-1)

        # Alpha-composite
        result_np = orig_np * (1.0 - mask_3ch) + relit_np * mask_3ch
        result_np = np.clip(result_np, 0, 255).astype(np.uint8)
        return Image.fromarray(result_np)
