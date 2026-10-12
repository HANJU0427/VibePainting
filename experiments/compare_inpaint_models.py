"""Reproducible, offline model comparison; production code and UI are untouched.

Run download_inpaint_models.py first. Use --models to select available checkpoints.
Outputs and timings are saved after each inference, so interrupted runs resume.
"""
import argparse
import gc
import hashlib
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Reuse the existing Windows optional-import workaround, without loading weights.
import sdxl_lightning_inpaint as engine
import diffusers
# Diffusers 0.30 probes ONNX classes even when loading PyTorch checkpoints.
# The shared Windows ONNX DLL cannot load; use Diffusers' own optional-backend
# placeholder for these type checks. No package files/environment are modified.
from diffusers.utils.dummy_onnx_objects import OnnxRuntimeModel as UnusedOnnxRuntimeModel
diffusers.OnnxRuntimeModel = UnusedOnnxRuntimeModel
import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw, ImageOps
from compare_control_fill import CASES, SEEDS, crop_box, font, render_tile

OUT = ROOT / 'experiments' / 'model_comparison'
PREVIOUS = ROOT / 'experiments' / 'fill_comparison'
MODELS = ROOT / 'models' / 'comparison'
NEGATIVE = 'low quality, bad anatomy, blurry'
CONFIG = {
    'lightning': dict(label='当前 Lightning', size=1024, steps=2, strength=1.0, cfg=0, controlnet_scale=.3),
    'sd15_inpaint': dict(label='SD 1.5 Inpainting', size=512, steps=30, strength=1.0, cfg=7.5),
    'sdxl_inpaint': dict(label='SDXL Inpainting', size=1024, steps=20, strength=.99, cfg=8.0),
    'lama': dict(label='LaMa', size=1024, steps=None, strength=None, cfg=None),
}


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def source(case):
    folder = PREVIOUS / case['id']
    return Image.open(folder / 'original.png').convert('RGB'), Image.open(folder / 'mask.png').convert('L')


def prepare(image, mask, size):
    fitted = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    x, y = (size-fitted.width)//2, (size-fitted.height)//2
    canvas = Image.new('RGB', (size, size), (127, 127, 127))
    canvas.paste(fitted, (x, y))
    work_mask = Image.new('L', (size, size))
    work_mask.paste(mask.resize(fitted.size, Image.Resampling.NEAREST), (x, y))
    # No access to the hidden original pixels, including SDXL's strength < 1 path.
    canvas = Image.composite(Image.new('RGB', canvas.size, 'white'), canvas, work_mask)
    return canvas, work_mask, (x, y, x+fitted.width, y+fitted.height)


def boundary_jump(image, mask):
    """RGB difference across adjacent pixels straddling the hard mask (0..255).

    A seam proxy only: true object edges, texture, blur and resolution affect it.
    """
    a = np.asarray(image).astype(np.float32)
    m = np.asarray(mask) > 0
    horizontal = np.abs(a[:, 1:] - a[:, :-1])[m[:, 1:] != m[:, :-1]].reshape(-1)
    vertical = np.abs(a[1:] - a[:-1])[m[1:] != m[:-1]].reshape(-1)
    return round(float(np.concatenate([horizontal, vertical]).mean()), 4)


def filename(model, kind, seed):
    return 'lama.png' if model == 'lama' else f'{model}_{kind}_seed{seed}.png'


def get_output(case, model, kind, seed):
    return OUT / case['id'] / filename(model, kind, seed)


def collect_baseline(records):
    old = json.loads((PREVIOUS / 'manifest.json').read_text(encoding='utf-8'))
    old_records = old.get('results', old.get('records'))
    for case in CASES:
        original, mask = source(case)
        folder = OUT / case['id']
        folder.mkdir(parents=True, exist_ok=True)
        original.save(folder / 'original.png')
        mask.save(folder / 'mask.png')
        for kind in ['empty', 'described']:
            for seed in SEEDS:
                dest = get_output(case, 'lightning', kind, seed)
                shutil.copyfile(PREVIOUS / case['id'] / f'{kind}_seed{seed}_white.png', dest)
                previous = next(r for r in old_records if r['case'] == case['id'] and r['prompt_kind'] == kind and r['seed'] == seed and r['fill'] == 'white')
                key = f'lightning/{case["id"]}/{kind}/{seed}'
                records[key] = dict(model='lightning', case=case['id'], prompt_kind=kind, seed=seed,
                                    seconds=previous['seconds'], reused_previous_run=True,
                                    file=str(dest.relative_to(OUT)).replace('\\', '/'),
                                    boundary_jump=boundary_jump(Image.open(dest), mask),
                                    original_boundary_jump=boundary_jump(original, mask),
                                    outside_mask_unchanged=previous['outside_mask_unchanged'])


def load_model(name):
    if name == 'lama':
        assert hashlib.md5((MODELS / 'lama' / 'big-lama.pt').read_bytes()).hexdigest() == 'e3aa4aaa15225a33ec84f9f4bc47e500', 'LaMa checkpoint checksum mismatch'
        return torch.jit.load(str(MODELS / 'lama' / 'big-lama.pt'), map_location='cuda').eval()
    cls = diffusers.StableDiffusionXLInpaintPipeline if name == 'sdxl_inpaint' else diffusers.StableDiffusionInpaintPipeline
    pipe = cls.from_pretrained(str(MODELS / name), torch_dtype=torch.float16,
                               variant='fp16', use_safetensors=True, local_files_only=True).to('cuda')
    assert pipe.unet.config.in_channels == 9, 'Expected a dedicated inpainting UNet'
    pipe.set_progress_bar_config(disable=True)
    return pipe


def infer(model, name, canvas, mask, prompt, seed):
    if name == 'lama':
        a = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1).copy() / 255
        m = (np.asarray(mask) > 0).astype(np.float32)[None].copy()
        result = model(torch.from_numpy(a)[None].to('cuda'), torch.from_numpy(m)[None].to('cuda'))
        arr = result[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)), None
    config = CONFIG[name]
    result = model(prompt=prompt, negative_prompt=NEGATIVE, image=canvas, mask_image=mask,
                   width=config['size'], height=config['size'],
                   num_inference_steps=config['steps'], strength=config['strength'],
                   guidance_scale=config['cfg'], generator=torch.Generator('cuda').manual_seed(seed))
    flags = getattr(result, 'nsfw_content_detected', None)
    return result.images[0], flags


def run(name, records, metadata):
    combinations = [('none', None)] if name == 'lama' else [(kind, seed) for kind in ['empty', 'described'] for seed in SEEDS]
    missing = [(c, k, s) for c in CASES for k, s in combinations if f'{name}/{c["id"]}/{k}/{s}' not in records]
    if not missing:
        return
    started = time.perf_counter()
    model = load_model(name)
    torch.cuda.synchronize()
    metadata.setdefault('loads', {})[name] = round(time.perf_counter()-started, 3)
    metadata.setdefault('schedulers', {})[name] = dict(model.scheduler.config) if name != 'lama' else None
    save_json(OUT / 'manifest.json', metadata)
    print(f'Loaded {name} in {metadata["loads"][name]:.2f}s', flush=True)
    # TorchScript profiling can optimize on the second call; warm up three times.
    original, mask = source(CASES[0])
    canvas, work_mask, _ = prepare(original, mask, CONFIG[name]['size'])
    warmup_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(3):
            infer(model, name, canvas, work_mask, '', 0)
    torch.cuda.synchronize()
    metadata.setdefault('warmups', {})[name] = dict(passes=3, seconds=round(time.perf_counter()-warmup_start, 3))
    for case, kind, seed in missing:
        original, mask = source(case)
        canvas, work_mask, extent = prepare(original, mask, CONFIG[name]['size'])
        canvas.save(OUT / case['id'] / f'input_{CONFIG[name]["size"]}.png')
        work_mask.save(OUT / case['id'] / f'mask_{CONFIG[name]["size"]}.png')
        prompt = case['prompt'] if kind == 'described' else ''
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            raw, safety = infer(model, name, canvas, work_mask, prompt, seed)
        torch.cuda.synchronize()
        restored = raw.crop(extent).resize(original.size, Image.Resampling.LANCZOS)
        result = Image.composite(restored, original, mask)
        seconds = time.perf_counter()-started
        outside = np.asarray(mask) == 0
        assert np.array_equal(np.asarray(result)[outside], np.asarray(original)[outside]), 'Unmasked pixels changed'
        target = get_output(case, name, kind, seed)
        if name == 'lama' and target.exists():
            assert np.array_equal(np.asarray(Image.open(target)), np.asarray(result)), 'LaMa rerun changed output'
        result.save(target)
        raw.save(target.with_stem(target.stem + '_raw'))
        key = f'{name}/{case["id"]}/{kind}/{seed}'
        records[key] = dict(model=name, case=case['id'], prompt_kind=kind, prompt=prompt, seed=seed,
                            seconds=round(seconds, 3), gpu_peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20),
                            file=str(target.relative_to(OUT)).replace('\\', '/'), nsfw_content_detected=safety,
                            boundary_jump=boundary_jump(result, mask), original_boundary_jump=boundary_jump(original, mask),
                            outside_mask_unchanged=True)
        save_json(OUT / 'results.json', records)
        print(f'{name}: {case["id"]} {kind} seed={seed}: {seconds:.3f}s, boundary jump={records[key]["boundary_jump"]:.2f}', flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def render():
    names = list(CONFIG)
    for kind in ['empty', 'described']:
        sheet = Image.new('RGB', (1420, 1500), '#f2f5f7')
        d = ImageDraw.Draw(sheet)
        d.text((24, 15), '局部补全模型对比 · 边缘与纹理', font=font(29), fill='#203949')
        d.text((24, 64), ('空提示词' if kind == 'empty' else '相同的描述性提示词') + ' · Seed 42 · 硬遮罩合成，无羽化 · LaMa 不接收文字', font=font(19), fill='#627481')
        for col, title in enumerate(['原图（局部）'] + [CONFIG[n]['label'] for n in names]):
            d.text((24+col*278, 106), title, font=font(19), fill='#203949')
        for row, case in enumerate(CASES):
            original, mask = source(case)
            bbox = crop_box(mask)
            y = 150 + row*327
            d.text((24, y), case['title'], font=font(20), fill='#203949')
            pictures = [original] + [Image.open(get_output(case, n, kind, 42)) for n in names]
            for col, picture in enumerate(pictures):
                render_tile(sheet, picture.crop(bbox), 24+col*278, y+37, 258)
        sheet.save(OUT / f'overview_{kind}.png')
    for case in CASES:
        original, mask = source(case)
        bbox = crop_box(mask)
        sheet = Image.new('RGB', (1420, 1470), '#f2f5f7')
        d = ImageDraw.Draw(sheet)
        d.text((24, 15), case['title'] + ' · 全部种子与提示词', font=font(28), fill='#203949')
        for col, title in enumerate(['原图（局部）'] + [CONFIG[n]['label'] for n in names]):
            d.text((24+col*278, 69), title, font=font(19), fill='#203949')
        for row, (kind, seed) in enumerate((k, s) for k in ['empty', 'described'] for s in SEEDS):
            y = 108+row*327
            d.text((24, y), f'{"空提示词" if kind == "empty" else "描述性提示词"} / seed {seed}', font=font(20), fill='#203949')
            for col, picture in enumerate([original] + [Image.open(get_output(case, n, kind, seed)) for n in names]):
                render_tile(sheet, picture.crop(bbox), 24+col*278, y+36, 258)
        sheet.save(OUT / case['id'] / 'all_models.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='*', default=['sd15_inpaint', 'sdxl_inpaint', 'lama'])
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'results.json'
    records = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    collect_baseline(records)
    metadata = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8')) if (OUT / 'manifest.json').exists() else {
        'config': CONFIG, 'cases': CASES, 'seeds': SEEDS, 'negative_prompt': NEGATIVE,
        'versions': {'torch': torch.__version__, 'diffusers': diffusers.__version__, 'transformers': transformers.__version__},
        'gpu': torch.cuda.get_device_name(), 'engine_sha256': hashlib.sha256((ROOT / 'sdxl_lightning_inpaint.py').read_bytes()).hexdigest(),
        'notes': ['Original-resolution masks are identical to the fill comparison.',
                  'White missing regions; no hidden original pixels supplied to new models.',
                  'Native working resolution differs: SD1.5 512; others 1024. Fit with gray padding, no stretching.',
                  'Original-resolution hard composite, no feathering. Raw model outputs retained.',
                  'Current Lightning outputs and timings reused from the previous verified white-fill experiment.',
                  'LaMa deterministic: one result per case reused in each prompt/seed comparison.',
                  'Timings exclude model load, warmup, preprocessing, PNG save; include inference, device sync, restore/composite.',
                  'Boundary jump is a seam proxy only; not a semantic/texture quality score.'],
    }
    save_json(OUT / 'results.json', records)
    if not args.render_only:
        for name in args.models:
            run(name, records, metadata)
    summary = {}
    for name in CONFIG:
        values = [r for r in records.values() if r['model'] == name]
        if values:
            summary[name] = dict(count=len(values), median_seconds=statistics.median(r['seconds'] for r in values),
                                 mean_boundary_jump=statistics.mean(r['boundary_jump'] for r in values),
                                 min_seconds=min(r['seconds'] for r in values), max_seconds=max(r['seconds'] for r in values),
                                 all_outside_unchanged=all(r['outside_mask_unchanged'] for r in values))
    save_json(OUT / 'summary.json', summary)
    save_json(OUT / 'manifest.json', metadata)
    if len(records) == 52:
        render()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
