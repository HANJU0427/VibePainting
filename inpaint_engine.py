"""Serialized inpainting runtime: at most one cached model on the GPU."""
import gc
import hashlib
import threading

# Reuse the Windows TensorFlow/sklearn optional-import workaround.
import sdxl_lightning_inpaint as lightning
import torch
import numpy as np
import diffusers
from diffusers.utils.dummy_onnx_objects import OnnxRuntimeModel
from PIL import Image, ImageOps
from model_catalog import ROOT, DEFAULT_MODEL, get_model, require_available

# Diffusers 0.30 probes ONNX type classes while loading PyTorch pipelines.
# This application does not use ONNX; the shared Windows ONNX DLL fails to load.
diffusers.OnnxRuntimeModel = OnnxRuntimeModel


class ModelManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.pipeline = None
        self.model_id = None

    def release(self):
        self.pipeline = None
        self.model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, model_id):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.float32
        if model_id == 'lightning':
            pipe = lightning.create_pipeline()
        elif model_id == 'lama':
            path = ROOT/'models/comparison/lama/big-lama.pt'
            if hashlib.md5(path.read_bytes()).hexdigest() != 'e3aa4aaa15225a33ec84f9f4bc47e500':
                raise ValueError('LaMa 模型文件校验失败，请重新下载')
            return torch.jit.load(str(path), map_location=device).eval()
        else:
            cls = diffusers.StableDiffusionInpaintPipeline if model_id == 'sd15_inpaint' else diffusers.StableDiffusionXLInpaintPipeline
            pipe = cls.from_pretrained(str(ROOT/'models/comparison'/model_id),
                torch_dtype=dtype, variant='fp16', use_safetensors=True, local_files_only=True).to(device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def generate(self, image, mask, prompt='', negative_prompt='', model=DEFAULT_MODEL,
                 strength=None, num_inference_steps=None, guidance_scale=None,
                 controlnet_conditioning_scale=None, seed=None, progress=None):
        spec = get_model(model)
        require_available(model)
        image, mask = image.convert('RGB'), mask.convert('L')
        if image.size != mask.size or mask.getbbox() is None:
            raise ValueError('请提供尺寸一致的原图与非空遮罩')
        size = spec['size']
        fitted = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
        x, y = (size-fitted.width)//2, (size-fitted.height)//2
        canvas = Image.new('RGB', (size, size), (127,127,127)); canvas.paste(fitted,(x,y))
        work_mask = Image.new('L', (size, size))
        work_mask.paste(mask.resize(fitted.size, Image.Resampling.NEAREST),(x,y))
        if work_mask.getbbox() is None:
            raise ValueError('遮罩区域过小，请加大笔刷后重试')
        notify = progress or (lambda _: None)
        with self.lock:
            try:
                if self.model_id != model or self.pipeline is None:
                    notify(f'正在切换并加载 {spec["label"]}…')
                    self.release()
                    self.pipeline = self.load(model)
                    self.model_id = model
                notify(f'正在使用 {spec["label"]} 生成…')
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                with torch.inference_mode():
                    if model == 'lama':
                        pixels = np.asarray(canvas, dtype=np.float32).transpose(2,0,1).copy()/255
                        missing = (np.asarray(work_mask)>0).astype(np.float32)[None].copy()
                        output = self.pipeline(torch.from_numpy(pixels)[None].to(device), torch.from_numpy(missing)[None].to(device))
                        array = output[0].permute(1,2,0).cpu().numpy()
                        raw = Image.fromarray(np.clip(array*255,0,255).astype(np.uint8))
                    else:
                        values = dict(strength=strength, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
                        values = {k: spec['defaults'][k] if v is None else v for k,v in values.items()}
                        if not 0 < values['strength'] <= 1 or int(values['strength']*values['num_inference_steps']) < 1:
                            raise ValueError('重绘强度 × 推理步数必须至少为 1')
                        extra = {}
                        if model == 'lightning':
                            hard_mask = work_mask.point(lambda v:255 if v else 0)
                            extra = dict(control_image=Image.composite(Image.new('RGB',canvas.size,'white'),canvas,hard_mask),
                                controlnet_conditioning_scale=spec['defaults']['controlnet_conditioning_scale'] if controlnet_conditioning_scale is None else controlnet_conditioning_scale)
                        output = self.pipeline(prompt=prompt, negative_prompt=negative_prompt, image=canvas,
                            mask_image=work_mask, width=size, height=size, **values, **extra,
                            generator=torch.Generator(device).manual_seed(seed) if seed is not None else None)
                        if any(getattr(output, 'nsfw_content_detected', None) or []):
                            raise ValueError('模型自带检测器未返回可用图片，请更换提示词或随机种子后再试')
                        raw = output.images[0]
                restored = raw.crop((x,y,x+fitted.width,y+fitted.height)).resize(image.size,Image.Resampling.LANCZOS)
                return Image.composite(restored,image,mask)
            except Exception:
                # Failed loads/inferences must not retain a broken model or GPU cache.
                self.release()
                raise


manager = ModelManager()
inpaint_image = manager.generate
