from diffusers import PixArtSigmaPipeline as pixart
import torch

pipe = pixart.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    torch_dtype=torch.float32
).to("cpu")

image = pipe(
    prompt="random scene with only partail green and black"
).images[0]

image.save("test-2.png")