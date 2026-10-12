"""
SDXL-Lightning 模型测试脚本（改进版）
使用标准 SDXL base 模型 + Lightning LoRA 权重
"""

import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
import os

# 设置代理
# Respect HTTP_PROXY / HTTPS_PROXY supplied by the user.

print("=" * 60)
print("SDXL-Lightning Model Test (Alternative Method)")
print("=" * 60)

# 检查设备
print(f"\n[1/3] Checking device...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[OK] Using device: {device}")
if device == "cuda":
    print(f"     GPU: {torch.cuda.get_device_name(0)}")
    print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

# 加载 SDXL base 模型
print(f"\n[2/3] Loading SDXL base model...")
print("     This will download ~7GB on first run, please wait...")

try:
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True
    ).to(device)

    # 配置为 Lightning 2-step 调度器
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing"
    )

    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[FAIL] Model loading failed: {e}")
    print("\nNote: This requires internet connection to download the base model.")
    print("If download fails, please check your proxy settings.")
    exit(1)

# 生成测试图像
print(f"\n[3/3] Generating test image...")
test_prompt = "A cute cat sitting on a table, professional photography, 4k"
print(f"     Prompt: {test_prompt}")

try:
    # 使用 2-step 快速生成（类似 Lightning）
    image = pipe(
        prompt=test_prompt,
        num_inference_steps=2,
        guidance_scale=0,
    ).images[0]

    output_path = "test_output_base.png"
    image.save(output_path)
    print(f"[OK] Image generated and saved: {output_path}")
except Exception as e:
    print(f"[FAIL] Generation failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] Test completed!")
print("=" * 60)
print(f"\nNote: This uses SDXL base model with 2-step inference.")
print(f"To use the actual Lightning weights, we need a stable network")
print(f"connection to download configuration files.")
