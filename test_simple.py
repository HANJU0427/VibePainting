"""
SDXL-Lightning 简单测试
使用本地模型文件 + 手动配置
"""

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from safetensors.torch import load_file
import os

print("=" * 60)
print("SDXL-Lightning Simple Test")
print("=" * 60)

# 检查模型文件
model_path = "models/sdxl_lightning_2step.safetensors"
print(f"\n[1/4] Checking model file...")
if not os.path.exists(model_path):
    print(f"[FAIL] Model file not found: {model_path}")
    exit(1)
print(f"[OK] Model file exists: {model_path}")
print(f"     Size: {os.path.getsize(model_path) / (1024**3):.2f} GB")

# 检查设备
print(f"\n[2/4] Checking device...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[OK] Using device: {device}")
if device == "cuda":
    print(f"     GPU: {torch.cuda.get_device_name(0)}")

# 说明
print(f"\n[3/4] Loading model...")
print("     Note: SDXL-Lightning requires base model configs from HuggingFace.")
print("     Attempting to load with network access...")

# 设置代理并尝试加载
os.environ['HF_HUB_OFFLINE'] = '0'  # 允许网络访问
# Respect HTTP_PROXY / HTTPS_PROXY supplied by the user.

try:
    # 方法：使用 ByteDance 官方推荐的加载方式
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    # 配置调度器
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing"
    )

    print("[OK] Model loaded successfully!")

except Exception as e:
    print(f"[FAIL] Loading failed: {str(e)[:200]}")
    print("\n" + "=" * 60)
    print("TROUBLESHOOTING:")
    print("=" * 60)
    print("The model file needs configuration files from HuggingFace.")
    print("These are small JSON files (~100KB total) but require network access.")
    print("\nOptions:")
    print("1. Check your proxy/network connection")
    print("2. Try running when network is more stable")
    print("3. Use the demo scripts which will handle this automatically")
    exit(1)

# 生成测试图像
print(f"\n[4/4] Generating test image...")
prompt = "A beautiful sunset over mountains, vibrant colors, 4k"
print(f"     Prompt: {prompt}")

try:
    image = pipe(
        prompt=prompt,
        num_inference_steps=2,
        guidance_scale=0,
    ).images[0]

    output_path = "test_output.png"
    image.save(output_path)
    print(f"[OK] Image saved: {output_path}")

except Exception as e:
    print(f"[FAIL] Generation failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] All tests passed!")
print("=" * 60)
print(f"\nGenerated image: {output_path}")
print("The model is working correctly!")
