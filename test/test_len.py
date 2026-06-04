import torch
from lens import LensPipeline

pipe = LensPipeline.from_pretrained(
    "microsoft/Lens", torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt="A cat holding a sign that says \"hello world\"",
    base_resolution=1440, aspect_ratio="1:1",
    num_inference_steps=20, guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("lens.png")
