"""
测试 SDXL-Lightning 模型是否能正常加载和运行
"""

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
import os

# 设置代理以下载配置文件
# Respect HTTP_PROXY / HTTPS_PROXY supplied by the user.

print("=" * 60)
print("SDXL-Lightning Model Test")
print("=" * 60)

# 1. 检查模型文件
model_path = "models/sdxl_lightning_2step.safetensors"
print(f"\n[1/5] Checking model file...")
if os.path.exists(model_path):
    size_gb = os.path.getsize(model_path) / (1024**3)
    print(f"[OK] Model file exists: {model_path}")
    print(f"     File size: {size_gb:.2f} GB")
else:
    print(f"[FAIL] Model file not found: {model_path}")
    exit(1)

# 2. 检查设备
print(f"\n[2/5] Checking device...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[OK] Using device: {device}")
if device == "cuda":
    print(f"     GPU: {torch.cuda.get_device_name(0)}")
    print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
else:
    print("     Warning: CUDA not available, using CPU (slower)")

# 3. 加载模型
print(f"\n[3/5] Loading model...")
try:
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing"
    )
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[FAIL] Model loading failed: {e}")
    exit(1)

# 4. 生成测试图像
print(f"\n[4/5] Generating test image...")
test_prompt = "A cute cat sitting on a table, professional photography"
print(f"     Prompt: {test_prompt}")

try:
    image = pipe(
        prompt=test_prompt,
        num_inference_steps=2,
        guidance_scale=0,
    ).images[0]
    print("[OK] Image generated successfully")
except Exception as e:
    print(f"[FAIL] Image generation failed: {e}")
    exit(1)

# 5. 保存图像
print(f"\n[5/5] Saving image...")
output_path = "test_output.png"
try:
    image.save(output_path)
    print(f"[OK] Image saved: {output_path}")
except Exception as e:
    print(f"[FAIL] Save failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] All tests passed! Model is working properly.")
print("=" * 60)
print(f"\nPlease check the generated test image: {output_path}")
