"""Model names, supported controls and local checkpoint availability (no ML imports)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = 'lightning'
MODELS = {
    'lightning': {
        'label': 'SDXL Lightning · 快速生成', 'size': 1024,
        'description': '原来的两步补全模型，速度快，适合快速尝试。',
        'parameter_hint': '默认 2 步、引导 0；引导大于 1 时负面提示词才生效。',
        'supports_prompt': True,
        'defaults': {'strength': 1.0, 'num_inference_steps': 2, 'guidance_scale': 0, 'controlnet_conditioning_scale': .3},
        'parameters': ['strength', 'num_inference_steps', 'guidance_scale', 'controlnet_conditioning_scale', 'seed', 'negative_prompt'],
    },
    'sd15_inpaint': {
        'label': 'SD 1.5 Inpainting · 上下文补全', 'size': 512,
        'description': '适合延续周围内容与修补背景。使用 512 工作分辨率，指定新物体的效果因图片而异。',
        'parameter_hint': '默认 30 步、引导 7.5、重绘强度 1。',
        'supports_prompt': True,
        'defaults': {'strength': 1.0, 'num_inference_steps': 30, 'guidance_scale': 7.5},
        'parameters': ['strength', 'num_inference_steps', 'guidance_scale', 'seed', 'negative_prompt'],
    },
    'sdxl_inpaint': {
        'label': 'SDXL Inpainting · 1024 补全', 'size': 1024,
        'description': '在 1024 工作分辨率下补全，保留原图作为初始内容，生成时间较长。',
        'parameter_hint': '默认 20 步、引导 8、强度 0.99。建议保持强度低于 1。',
        'supports_prompt': True,
        'defaults': {'strength': .99, 'num_inference_steps': 20, 'guidance_scale': 8.0},
        'parameters': ['strength', 'num_inference_steps', 'guidance_scale', 'seed', 'negative_prompt'],
    },
    'lama': {
        'label': 'LaMa · 快速修补 / 擦除', 'size': 1024,
        'description': '自动延续背景、修补或擦除物体；无需提示词，不支持指定新内容。',
        'parameter_hint': '固定修补模式，无需采样参数或随机种子。',
        'supports_prompt': False, 'defaults': {}, 'parameters': [],
    },
}


def get_model(model_id):
    if not isinstance(model_id, str) or model_id not in MODELS:
        raise ValueError('不支持的模型，请刷新页面后重新选择')
    return MODELS[model_id]


def model_files(model_id):
    get_model(model_id)
    if model_id == 'lightning':
        return [ROOT/'models/sdxl_lightning_2step.safetensors',
                ROOT/'models/controlnet-inpaint-sdxl/config.json',
                ROOT/'models/controlnet-inpaint-sdxl/diffusion_pytorch_model.safetensors']
    if model_id == 'lama':
        return [ROOT/'models/comparison/lama/big-lama.pt']
    root = ROOT/'models/comparison'/model_id
    files = ['model_index.json', 'scheduler/scheduler_config.json', 'unet/config.json',
             'unet/diffusion_pytorch_model.fp16.safetensors', 'vae/config.json',
             'vae/diffusion_pytorch_model.fp16.safetensors', 'text_encoder/config.json',
             'text_encoder/model.fp16.safetensors', 'tokenizer/vocab.json', 'tokenizer/merges.txt',
             'tokenizer/tokenizer_config.json', 'tokenizer/special_tokens_map.json']
    if model_id == 'sdxl_inpaint':
        files += ['text_encoder_2/config.json', 'text_encoder_2/model.fp16.safetensors',
                  'tokenizer_2/vocab.json', 'tokenizer_2/merges.txt',
                  'tokenizer_2/tokenizer_config.json', 'tokenizer_2/special_tokens_map.json']
    else:
        files += ['safety_checker/config.json', 'safety_checker/model.fp16.safetensors',
                  'feature_extractor/preprocessor_config.json']
    return [root/path for path in files]


def require_available(model_id):
    if not all(p.is_file() for p in model_files(model_id)):
        raise ValueError(f'{get_model(model_id)["label"]} 的本地模型文件不完整，请先准备模型文件')


def list_models():
    return [{'id': key, **spec, 'available': all(p.is_file() for p in model_files(key))}
            for key, spec in MODELS.items()]
