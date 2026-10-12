"""
SDXL-Lightning 简单使用脚本
快速生成图像的简化版本
"""

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
import os

# 检查模型文件是否存在
model_path = "models/sdxl_lightning_2step.safetensors"
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    print("请先下载模型文件，参考 models/README.md")
    exit(1)

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载模型
print("正在加载 SDXL-Lightning 模型...")
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

# 配置调度器
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    timestep_spacing="trailing"
)

print("模型加载完成！\n")

# 交互式生成
while True:
    prompt = input("请输入提示词 (输入 'quit' 退出): ")

    if prompt.lower() == 'quit':
        break

    if not prompt.strip():
        continue

    print(f"\n正在生成图像...")

    # 生成图像
    image = pipe(
        prompt=prompt,
        num_inference_steps=2,  # Lightning 2-step 模型
        guidance_scale=0,       # Lightning 推荐设置
    ).images[0]

    # 保存图像
    output_path = f"output_{hash(prompt) % 10000}.png"
    image.save(output_path)
    print(f"✓ 图像已保存: {output_path}\n")

print("再见！")
