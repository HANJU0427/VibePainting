"""
SDXL-Lightning 2-step 模型使用示例
这个脚本展示如何使用 SDXL-Lightning 模型生成图像
"""

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

def generate_image(
    prompt,
    output_path="output.png",
    num_inference_steps=2,
    guidance_scale=0,
    seed=None
):
    """
    使用 SDXL-Lightning 生成图像

    参数:
        prompt: 文本提示词
        output_path: 输出图像路径
        num_inference_steps: 推理步数（2步模型固定为2）
        guidance_scale: 引导系数（Lightning模型推荐为0）
        seed: 随机种子（可选）
    """

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载模型
    print("正在加载模型...")
    model_path = "models/sdxl_lightning_2step.safetensors"

    # 使用 SDXL base 作为基础，加载 Lightning 权重
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    # 设置调度器为 Euler
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
    print(f"正在生成图像: {prompt}")
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    # 保存图像
    image.save(output_path)
    print(f"图像已保存到: {output_path}")

    return image


if __name__ == "__main__":
    # 示例1: 基础使用
    prompt = "A man standing on the left, and a woman sitting on the right"
    generate_image(prompt, output_path="output_landscape.png", seed=42)

    # 示例2: 人物肖像
    # prompt2 = "Portrait of a young woman with blue eyes, professional photography, studio lighting"
    # generate_image(prompt2, output_path="output_portrait.png", seed=123)

    print("\n生成完成！")
