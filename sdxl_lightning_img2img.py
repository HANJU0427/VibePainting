"""
SDXL-Lightning Image-to-Image 示例
支持输入图像作为提示，进行图像转换、风格迁移等
"""

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler
from PIL import Image
import os

def generate_from_image(
    input_image_path,
    prompt,
    output_path="output_img2img.png",
    strength=0.75,
    num_inference_steps=2,
    guidance_scale=0,
    seed=None
):
    """
    使用 SDXL-Lightning 进行图像到图像的转换

    参数:
        input_image_path: 输入图像路径
        prompt: 文本提示词，描述想要的效果
        output_path: 输出图像路径
        strength: 转换强度 (0.0-1.0)，越高变化越大
                 0.3-0.5: 轻微修改
                 0.5-0.7: 中等修改
                 0.7-1.0: 大幅修改
        num_inference_steps: 推理步数（Lightning 2-step 固定为 2）
        guidance_scale: 引导系数（Lightning 推荐为 0）
        seed: 随机种子（可选）
    """

    # 检查输入图像
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"输入图像不存在: {input_image_path}")

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载输入图像
    print(f"加载输入图像: {input_image_path}")
    init_image = Image.open(input_image_path).convert("RGB")

    # 调整图像大小为 1024x1024（SDXL 标准尺寸）
    init_image = init_image.resize((1024, 1024))
    print(f"图像尺寸: {init_image.size}")

    # 加载模型
    print("正在加载 SDXL-Lightning 模型...")
    model_path = "models/sdxl_lightning_2step.safetensors"

    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    # 设置调度器
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing"
    )

    print("模型加载完成！")

    # 设置随机种子
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    # 生成图像
    print(f"正在生成图像...")
    print(f"  提示词: {prompt}")
    print(f"  强度: {strength}")

    image = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    # 保存图像
    image.save(output_path)
    print(f"图像已保存到: {output_path}")

    return image


if __name__ == "__main__":
    # 示例：需要先有一张输入图像
    # 你可以修改下面的路径和提示词

    input_image = "input.jpg"  # 修改为你的输入图像路径

    if not os.path.exists(input_image):
        print(f"错误: 输入图像不存在: {input_image}")
        print("\n使用方法:")
        print("1. 准备一张输入图像，命名为 input.jpg")
        print("2. 或者修改脚本中的 input_image 变量为你的图像路径")
        print("3. 运行此脚本")
        print("\n示例:")
        print('  generate_from_image("my_photo.jpg", "anime style, vibrant colors")')
    else:
        # 示例1: 转换为动漫风格
        generate_from_image(
            input_image_path=input_image,
            prompt="anime style, vibrant colors, 2D illustration",
            output_path="output_anime.png",
            strength=0.7,
            seed=42
        )

        # 示例2: 转换为油画风格
        generate_from_image(
            input_image_path=input_image,
            prompt="oil painting style, impressionist, artistic",
            output_path="output_painting.png",
            strength=0.6,
            seed=123
        )

        print("\n生成完成！")
