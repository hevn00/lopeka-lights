import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        from diffusers import StableDiffusionImg2ImgPipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stablediffusionapi/realistic-vision-v51",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

    def predict(
        self,
        image_url: str = Input(description="Kaynak resim URL"),
        light_direction: float = Input(description="Işık yönü 0-360", default=0),
        light_height: float = Input(description="Işık yüksekliği 0-1", default=0.5),
        prompt: str = Input(description="Ek prompt", default=""),
    ) -> Path:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize((512, 512))

        angle = light_direction
        if 315 <= angle or angle < 45:
            dir_text = "from the right side"
        elif 45 <= angle < 135:
            dir_text = "from above"
        elif 135 <= angle < 225:
            dir_text = "from the left side"
        else:
            dir_text = "from below"

        height_text = "high" if light_height > 0.6 else "low" if light_height < 0.4 else "medium"
        light_prompt = f"same scene, same subject, cinematic lighting {dir_text}, {height_text} angle light, dramatic shadows, photorealistic"
        if prompt:
            light_prompt += f", {prompt}"

        result = self.pipe(
            prompt=light_prompt,
            image=image,
            strength=0.35,
            guidance_scale=7.5,
            num_inference_steps=30,
        ).images[0]

        output_path = "/tmp/output.png"
        result.save(output_path)
        return Path(output_path)
