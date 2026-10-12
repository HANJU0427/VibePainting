# SDXL-Lightning 使用指南

> 本文记录早期文生图 / 图生图 CLI。当前网页工作台、依赖版本和模型下载步骤以 [README](README.md) 为准。

## 项目结构
```
VibePainting/
├── models/
│   └── sdxl_lightning_2step.safetensors  # 模型文件（需手动下载）
├── sdxl_lightning_demo.py                 # 完整示例脚本
├── quick_generate.py                      # 快速交互式生成脚本
└── USAGE.md                               # 本文件
```

## 安装依赖

请按 [README 的安装步骤](README.md#1-克隆并安装依赖) 建立独立虚拟环境，并使用 `requirements.txt` 中的兼容版本。执行下文 CLI 命令前，可先运行 `.\.venv\Scripts\Activate.ps1` 激活环境，或将 `python` 替换为 `.\.venv\Scripts\python.exe`。

## 下载模型

模型文件需要另外下载，详见 [README 的模型准备步骤](README.md#2-准备模型)。

**重要**: 确保模型文件路径为 `models/sdxl_lightning_2step.safetensors`

## 使用方法

### 方法1: 文本生成图像（Text-to-Image）

#### 快速交互式生成（推荐新手）

运行交互式脚本：
```bash
python quick_generate.py
```

然后输入提示词，例如：
```
请输入提示词: A beautiful sunset over the ocean, vibrant colors, 4k
```

程序会自动生成图像并保存。

#### 使用示例脚本

运行完整示例：
```bash
python sdxl_lightning_demo.py
```

这会生成示例图像。

#### 在代码中使用

```python
from sdxl_lightning_demo import generate_image

# 生成图像
generate_image(
    prompt="Your prompt here",
    output_path="my_image.png",
    seed=42  # 可选，用于可重复的结果
)
```

### 方法2: 图像生成图像（Image-to-Image）

SDXL-Lightning 支持输入图像作为基础，进行风格转换、修改等操作。

#### 快速交互式转换

运行交互式脚本：
```bash
python quick_img2img.py
```

然后按提示输入：
1. 输入图像路径（如 `input.jpg`）
2. 提示词（如 `anime style, vibrant colors`）
3. 转换强度（0.0-1.0，推荐 0.5-0.7）

#### 使用示例脚本

```bash
python sdxl_lightning_img2img.py
```

需要先准备一张输入图像命名为 `input.jpg`。

#### 在代码中使用

```python
from sdxl_lightning_img2img import generate_from_image

# 图像转换
generate_from_image(
    input_image_path="my_photo.jpg",
    prompt="anime style, vibrant colors",
    output_path="output.png",
    strength=0.7,  # 转换强度
    seed=42
)
```

#### 转换强度说明

- **0.3-0.5**: 轻微修改，保留原图大部分特征
- **0.5-0.7**: 中等修改，平衡原图和新风格
- **0.7-1.0**: 大幅修改，更接近完全重新生成

## 提示词建议

### Text-to-Image 提示词示例：
- `"A serene mountain landscape at dawn, misty valleys, golden light, highly detailed"`
- `"Portrait of a cyberpunk character, neon lights, futuristic city background, 4k"`
- `"A cute cat sitting on a windowsill, soft natural lighting, photorealistic"`
- `"Abstract art with vibrant colors, geometric shapes, modern style"`

### Image-to-Image 提示词示例：
- `"anime style, vibrant colors, 2D illustration"` - 转换为动漫风格
- `"oil painting style, impressionist, artistic"` - 转换为油画风格
- `"watercolor painting, soft colors, artistic"` - 水彩画风格
- `"cyberpunk style, neon lights, futuristic"` - 赛博朋克风格
- `"black and white, high contrast, dramatic lighting"` - 黑白高对比度
- `"vintage photo, sepia tone, 1920s style"` - 复古照片风格

### 提示词技巧：
1. **描述清晰**: 包含主体、环境、光照、风格等
2. **添加质量词**: 如 "highly detailed", "4k", "professional photography"
3. **指定风格**: 如 "oil painting", "digital art", "photorealistic"
4. **使用英文**: 模型对英文提示词效果更好

## 参数说明

### `generate_image()` 函数参数：

- **prompt** (str): 文本提示词，描述你想生成的图像
- **output_path** (str): 输出图像的保存路径，默认 "output.png"
- **num_inference_steps** (int): 推理步数，2-step 模型固定为 2
- **guidance_scale** (float): 引导系数，Lightning 模型推荐为 0
- **seed** (int, 可选): 随机种子，用于生成可重复的结果

## 性能说明

当前使用 Lightning 2-step 权重。耗时和显存会受模型、分辨率、首次加载及硬件影响；CLI 文生图的结果不能代替带 ControlNet 的网页补全测试。已验证环境与当前限制见 [README](README.md)，其他 GPU 与 CPU 的性能尚未完整验证。

## 常见问题

### Q: 提示 "CUDA out of memory"
A: 显存不足，尝试：
```python
pipe.enable_attention_slicing()  # 在加载模型后添加
```

### Q: 生成的图像质量不好
A: 尝试：
1. 改进提示词，添加更多细节
2. 使用不同的随机种子
3. 考虑使用 4-step 或 8-step 模型（需重新下载）

### Q: 模型加载很慢
A: 首次加载需要时间，后续会快很多。可以考虑：
```python
pipe.enable_model_cpu_offload()  # 节省显存但会稍慢
```

### Q: 想要更高质量的图像
A: 可以下载 4-step 或 8-step 版本：
- `sdxl_lightning_4step.safetensors` - 4步推理，质量更好
- `sdxl_lightning_8step.safetensors` - 8步推理，质量最好

## 进阶使用

### 批量生成
```python
prompts = [
    "A cat in a garden",
    "A dog on a beach",
    "A bird in the sky"
]

for i, prompt in enumerate(prompts):
    generate_image(prompt, f"output_{i}.png", seed=i)
```

### 固定风格
```python
style = "oil painting, impressionist style, vibrant colors"
prompt = f"A sunset over mountains, {style}"
generate_image(prompt)
```

### 图像尺寸调整
默认生成 1024x1024，如需其他尺寸，修改代码：
```python
image = pipe(
    prompt=prompt,
    num_inference_steps=2,
    guidance_scale=0,
    height=768,  # 自定义高度
    width=1024   # 自定义宽度
).images[0]
```

## 许可证

SDXL-Lightning 使用 OpenRAIL++ 许可证。详见：
https://huggingface.co/ByteDance/SDXL-Lightning

## 参考资源

- 官方仓库: https://huggingface.co/ByteDance/SDXL-Lightning
- 论文: https://arxiv.org/abs/2402.13929
- Diffusers 文档: https://huggingface.co/docs/diffusers

## 技术支持

如遇问题，请检查：
1. 模型文件是否正确下载并放置在 `models/` 文件夹
2. 依赖库是否正确安装
3. GPU 驱动是否最新（如使用 GPU）
