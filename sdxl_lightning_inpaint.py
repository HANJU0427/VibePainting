"""
SDXL-Lightning + ControlNet Inpainting
使用 SDXL-Lightning 作为基础模型，结合 ControlNet 实现 inpainting 功能

用法:
    python sdxl_lightning_inpaint.py

需要准备:
    1. 输入图像 (RGB)
    2. 遮罩图像 (黑白，白色区域为需要重绘的部分)
    3. 文本提示词
"""

import os
from pathlib import Path
from threading import Lock

# This application only uses PyTorch. Do not import unrelated TensorFlow DLLs.
os.environ.setdefault('USE_TF', '0')

import torch
import transformers.utils.import_utils as transformers_imports

# Image-only runtime workaround for this shared Windows environment:
# Transformers' optional text-generation path imports sklearn -> pandas ->
# pyarrow, whose native import has crashed with access violations/stack overflow.
# CLIP image conditioning does not use sklearn's candidate-generation features.
# This availability flag is used by the installed Transformers 4.x release;
# do not uninstall or modify packages shared with other projects.
if hasattr(transformers_imports, '_sklearn_available'):
    transformers_imports._sklearn_available = False

from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, EulerDiscreteScheduler
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
_pipe = None
_pipeline_lock = Lock()


def get_pipeline():
    """Load once; callers serialize inference with _pipeline_lock."""
    global _pipe
    if _pipe is None:
        _pipe = create_pipeline()
    return _pipe


def create_pipeline():
    """Build an uncached pipeline so the web model manager owns its lifetime."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        str(ROOT / 'models/controlnet-inpaint-sdxl'), torch_dtype=dtype,
    )
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
        str(ROOT / 'models/sdxl_lightning_2step.safetensors'),
        controlnet=controlnet, torch_dtype=dtype,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing='trailing', prediction_type='epsilon',
    )
    return pipe


def inpaint_image(image, mask, prompt, negative_prompt='low quality, bad anatomy, blurry',
                  strength=1.0, num_inference_steps=2, guidance_scale=0,
                  controlnet_conditioning_scale=0.3, seed=None, progress=None):
    """In-memory entry point shared by the web page and the original script.

    Fit inside a square without stretching; composite at original resolution so
    pixels outside the supplied mask remain exactly unchanged.
    """
    image = image.convert('RGB')
    mask = mask.convert('L')
    if image.size != mask.size:
        raise ValueError('原图和遮罩尺寸必须一致')
    if mask.getbbox() is None:
        raise ValueError('请先绘制需要补全的区域')
    if not 0 < strength <= 1 or int(num_inference_steps * strength) < 1:
        raise ValueError('重绘强度 × 推理步数必须至少为 1')
    fitted = ImageOps.contain(image, (1024, 1024), Image.Resampling.LANCZOS)
    x, y = (1024 - fitted.width) // 2, (1024 - fitted.height) // 2
    init_image = Image.new('RGB', (1024, 1024), (127, 127, 127))
    init_image.paste(fitted, (x, y))
    mask_image = Image.new('L', (1024, 1024), 0)
    mask_image.paste(mask.resize(fitted.size, Image.Resampling.NEAREST), (x, y))
    if mask_image.getbbox() is None:
        raise ValueError('遮罩区域过小，请加大笔刷后重试')
    # The Dreamer ControlNet was trained with solid-white missing regions.
    # https://huggingface.co/destitech/controlnet-inpaint-dreamer-sdxl
    condition_mask = mask_image.point(lambda value: 255 if value else 0)
    control_image = Image.composite(Image.new('RGB', (1024, 1024), 'white'), init_image, condition_mask)
    with _pipeline_lock:
        if progress:
            progress('正在加载模型（首次可能需要下载配置）…' if _pipe is None else '正在准备生成…')
        pipe = get_pipeline()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        if progress:
            progress('正在生成补全…')
        with torch.inference_mode():
            result = pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                image=init_image, mask_image=mask_image, control_image=control_image,
                strength=strength, num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
            ).images[0]
    result = result.crop((x, y, x + fitted.width, y + fitted.height))
    result = result.resize(image.size, Image.Resampling.LANCZOS)
    return Image.composite(result, image, mask)


def inpaint(
    image_path,
    mask_path,
    prompt,
    output_path="output_inpaint.png",
    negative_prompt="low quality, bad anatomy, blurry",
    strength=1.0,
    num_inference_steps=2,
    guidance_scale=0,
    controlnet_conditioning_scale=0.3,
    seed=None
):
    """
    使用 SDXL-Lightning + ControlNet 进行 inpainting

    参数:
        image_path: 输入图像路径
        mask_path: 遮罩图像路径 (白色=重绘区域, 黑色=保留区域)
        prompt: 文本提示词，描述重绘区域内容
        output_path: 输出图像路径
        negative_prompt: 负面提示词
        strength: 重绘强度 (0.0-1.0)
        num_inference_steps: 推理步数 (当前 Lightning 2-step 权重默认 2 步)
        guidance_scale: 引导系数 (Lightning 默认 0)
        controlnet_conditioning_scale: ControlNet 条件强度 (0.0-1.0)
        seed: 随机种子
    """

    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        result = inpaint_image(
            image, mask, prompt, negative_prompt, strength, num_inference_steps,
            guidance_scale, controlnet_conditioning_scale, seed, progress=print,
        )

    result.save(output_path)
    print(f"Result saved: {output_path}")

    return result


def create_test_mask(image_path, output_mask_path="test_mask.png"):
    """创建一个简单的测试遮罩（中心 1/3 区域为白色）"""
    img = Image.open(image_path)
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    x1, y1 = w // 3, h // 3
    x2, y2 = 2 * w // 3, 2 * h // 3
    draw.rectangle([x1, y1, x2, y2], fill=255)
    mask.save(output_mask_path)
    print(f"Test mask saved: {output_mask_path}")
    return output_mask_path


if __name__ == "__main__":
    input_image = "input.jpg"

    if not os.path.exists(input_image):
        if os.path.exists("output_landscape.png"):
            input_image = "output_landscape.png"
        else:
            print("Please provide an input image (input.jpg or output_landscape.png)")
            exit(1)

    # 创建测试遮罩
    mask_path = create_test_mask(input_image)

    # 执行 inpainting
    inpaint(
        image_path=input_image,
        mask_path=mask_path,
        prompt="a beautiful garden with colorful flowers and green leaves, natural, detailed",
        output_path="output_inpaint.png",
        seed=42
    )

    print("\nDone!")
