"""
SDXL-Lightning 交互式 Image-to-Image 工具
支持输入图像进行风格转换、修改等操作
"""

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler
from PIL import Image
import os

# 设置代理（如果需要）
# Respect HTTP_PROXY / HTTPS_PROXY supplied by the user.

# 检查模型文件
model_path = "models/sdxl_lightning_2step.safetensors"
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    print("请先下载模型文件，参考 models/README.md")
    exit(1)

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载模型
print("正在加载 SDXL-Lightning Image-to-Image 模型...")
pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

# 配置调度器
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    timestep_spacing="trailing"
)

print("模型加载完成！\n")

print("=" * 60)
print("SDXL-Lightning Image-to-Image 交互式工具")
print("=" * 60)
print("\n使用说明:")
print("1. 输入图像路径")
print("2. 输入提示词（描述想要的效果）")
print("3. 输入转换强度（0.0-1.0，推荐 0.5-0.7）")
print("4. 等待生成完成")
print("\n输入 'quit' 退出\n")

while True:
    # 获取输入图像路径
    image_path = input("输入图像路径: ").strip()

    if image_path.lower() == 'quit':
        break

    if not image_path:
        continue

    # 移除可能的引号
    image_path = image_path.strip('"').strip("'")

    if not os.path.exists(image_path):
        print(f"错误: 文件不存在: {image_path}\n")
        continue

    # 获取提示词
    prompt = input("提示词 (描述想要的效果): ").strip()

    if not prompt:
        print("错误: 提示词不能为空\n")
        continue

    # 获取强度
    strength_input = input("转换强度 (0.0-1.0, 默认 0.7): ").strip()
    try:
        strength = float(strength_input) if strength_input else 0.7
        strength = max(0.0, min(1.0, strength))  # 限制在 0-1 之间
    except ValueError:
        print("使用默认强度: 0.7")
        strength = 0.7

    print(f"\n正在处理...")
    print(f"  输入图像: {image_path}")
    print(f"  提示词: {prompt}")
    print(f"  强度: {strength}")

    try:
        # 加载并调整图像
        init_image = Image.open(image_path).convert("RGB")
        original_size = init_image.size
        init_image = init_image.resize((1024, 1024))

        # 生成图像
        image = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=2,
            guidance_scale=0,
        ).images[0]

        # 保存图像
        output_path = f"output_img2img_{hash(prompt) % 10000}.png"
        image.save(output_path)
        print(f"✓ 图像已保存: {output_path}")
        print(f"  原始尺寸: {original_size}, 输出尺寸: {image.size}\n")

    except Exception as e:
        print(f"✗ 生成失败: {e}\n")

print("再见！")
