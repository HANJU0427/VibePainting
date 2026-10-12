# SDXL-Lightning 功能总结

> 这是早期 CLI 功能记录；当前网页画笔、局部补全、多模型切换及已验证环境见 [README](README.md)。下列速度与显存数字仅对应当时的 CLI 试验，不适用于完整网页补全流程。

## ✅ 支持的功能

### 1. Text-to-Image (文本生成图像)
从文本描述生成全新的图像。

**脚本:**
- `quick_generate.py` - 交互式生成
- `sdxl_lightning_demo.py` - 示例脚本

**使用场景:**
- 创作原创艺术作品
- 概念设计
- 插画生成
- 任何从零开始的图像创作

### 2. Image-to-Image (图像生成图像) ✨
基于输入图像进行转换和修改。

**脚本:**
- `quick_img2img.py` - 交互式转换
- `sdxl_lightning_img2img.py` - 示例脚本

**使用场景:**
- 风格转换（照片→动漫、油画等）
- 图像增强和修改
- 艺术风格迁移
- 图像重绘和优化

**转换强度控制:**
- 0.3-0.5: 轻微修改
- 0.5-0.7: 中等修改（推荐）
- 0.7-1.0: 大幅修改

## 📊 模型特点

- **超快速度**: 仅需 2 步推理
- **高质量**: 接近标准 SDXL 的图像质量
- **低显存**: 约 6-8GB 显存（GPU）
- **灵活性**: 支持 text-to-image 和 image-to-image

## 🎯 快速开始

### Text-to-Image
```bash
python quick_generate.py
```

### Image-to-Image
```bash
python quick_img2img.py
```

## 📝 代码示例

### Text-to-Image
```python
from sdxl_lightning_demo import generate_image

generate_image(
    prompt="A beautiful landscape",
    output_path="output.png"
)
```

### Image-to-Image
```python
from sdxl_lightning_img2img import generate_from_image

generate_from_image(
    input_image_path="input.jpg",
    prompt="anime style, vibrant colors",
    output_path="output.png",
    strength=0.7
)
```

## 🚀 性能

**RTX 4090:**
- Text-to-Image: ~1-2 秒/张
- Image-to-Image: ~1-2 秒/张

## 📚 完整文档

详见 [USAGE.md](USAGE.md)
