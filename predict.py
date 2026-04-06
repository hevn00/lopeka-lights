import torch
import subprocess
from PIL import Image
from cog import BasePredictor, Input, Path

from diffusers import StableDiffusionImg2ImgPipeline


class Predictor(BasePredictor):
    def setup(self):
        print("[setup] Loading SD 1.5 img2img pipeline...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to("cuda")
        print("[setup] Done.")

    def predict(
        self,
        image_url: str = Input(description="Kaynak resim URL"),
        light_direction: float = Input(description="Işık yönü 0-360", default=0.0, ge=0.0, le=360.0),
        light_height: float = Input(description="Işık yüksekliği 0-1", default=0.5, ge=0.0, le=1.0),
        prompt: str = Input(description="Ek prompt", default=""),
    ) -> Path:
        # Download image
        print(f"[predict] Downloading: {image_url}")
        r = subprocess.run(["curl", "-s", "-L", "-o", "/tmp/input.jpg", image_url], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"Download failed: {r.stderr.decode()}")

        image = Image.open("/tmp/input.jpg").convert("RGB").resize((512, 512))

        # Build direction text
        angle = light_direction
        if 315 <= angle or angle < 45:
            dir_text = "from the right side"
        elif 45 <= angle < 135:
            dir_text = "from above"
        elif 135 <= angle < 225:
            dir_text = "from the left side"
        else:
            dir_text = "from below"

        height_text = "high angle" if light_height > 0.6 else "low angle" if light_height < 0.4 else "medium angle"

        light_prompt = (
            f"same scene, same subject, cinematic lighting {dir_text}, "
            f"{height_text} light, dramatic shadows, photorealistic, high quality"
        )
        if prompt:
            light_prompt += f", {prompt}"

        print(f"[predict] Prompt: {light_prompt}")

        result = self.pipe(
            prompt=light_prompt,
            image=image,
            strength=0.4,
            guidance_scale=8.0,
            num_inference_steps=30,
        ).images[0]

        out = "/tmp/output.png"
        result.save(out)
        print("[predict] Done.")
        return Path(out)
