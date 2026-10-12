import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['DIFFUSERS_ATTENTION_BACKEND'] = 'xformers'

import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("ByteDance/SDXL-Lightning", dtype=torch.bfloat16, device_map="cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]