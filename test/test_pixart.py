from diffusers import PixArtSigmaPipeline as pixart
import torch

seed = 45
generator = torch.Generator(device="cpu").manual_seed(seed)

pipe = pixart.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    torch_dtype=torch.float32
).to("cpu")

image = pipe(
    prompt="a landscape with partial blue and white tones",
    negative_prompt="blue",
    num_inference_steps=12,
    generator=generator,
    guidance_scale=1.5
).images[0]

image.save("prompt_3.png")