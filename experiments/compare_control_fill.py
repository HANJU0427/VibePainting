"""Paired local SDXL experiment; does not modify the web UI or production engine.

Run with the project's GPU Python. Stop the web inference server first to free
VRAM, and start it again after this script exits.
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sdxl_lightning_inpaint as engine
import numpy as np
import torch
import diffusers
import transformers
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

OUT = ROOT / 'experiments' / 'fill_comparison'
SEEDS = [42, 123]
RING = 24
SCALE = 0.3
CASES = [
    dict(id='sky', title='天空背景', source='111.jpg', box=(.16, .07, .54, .21),
         prompt='a pale blue sky with soft white clouds, natural daylight, photograph'),
    dict(id='yellow', title='黄色卡通表面', source='222.jpg', box=(.66, .45, .88, .70),
         prompt='a smooth yellow cartoon creature body, soft studio lighting, 3D render'),
    dict(id='bird', title='鸟的羽毛与轮廓', source='output_img2img_7513.png', box=(.40, .50, .67, .73),
         prompt='an orange bird chest with detailed soft feathers, wildlife photograph'),
    dict(id='corridor', title='建筑走廊 / 移除行人', source='output_989.png', box=(.42, .50, .61, .88),
         prompt='an empty modern concrete corridor, continuous floor, natural sunlight, architectural photograph'),
]


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size)


def crop_box(mask):
    x0, y0, x1, y1 = mask.getbbox()
    size = int(max(x1-x0, y1-y0) * 1.5)
    cx, cy = (x0+x1)//2, (y0+y1)//2
    # Fit a square in the original image, adding context without distortion.
    size = min(size, mask.width, mask.height)
    x0 = max(0, min(mask.width-size, cx-size//2))
    y0 = max(0, min(mask.height-size, cy-size//2))
    return x0, y0, x0+size, y0+size


def prepare(case):
    image = Image.open(ROOT / case['source']).convert('RGB')
    mask = Image.new('L', image.size)
    box = tuple(round(value * image.size[i % 2]) for i, value in enumerate(case['box']))
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    fitted = ImageOps.contain(image, (1024, 1024), Image.Resampling.LANCZOS)
    x, y = (1024-fitted.width)//2, (1024-fitted.height)//2
    init = Image.new('RGB', (1024, 1024), (127, 127, 127))
    init.paste(fitted, (x, y))
    work_mask = Image.new('L', (1024, 1024))
    work_mask.paste(mask.resize(fitted.size, Image.Resampling.NEAREST), (x, y))
    hard_mask = work_mask.point(lambda p: 255 if p else 0)
    valid = np.zeros((1024, 1024), dtype=bool)
    valid[y:y+fitted.height, x:x+fitted.width] = True
    inside = np.asarray(hard_mask) > 0
    dilated = np.asarray(hard_mask.filter(ImageFilter.MaxFilter(RING*2+1))) > 0
    ring = dilated & ~inside & valid
    assert ring.any()
    mean_float = np.asarray(init)[ring].mean(axis=0)
    mean = tuple(np.rint(mean_float).astype(int).tolist())
    controls = {name: Image.composite(Image.new('RGB', (1024, 1024), color), init, hard_mask)
                for name, color in [('white', (255, 255, 255)), ('mean', mean)]}
    return dict(image=image, mask=mask, init=init, work_mask=work_mask, controls=controls,
                extent=(x, y, x+fitted.width, y+fitted.height), mean=mean,
                ring_pixels=int(ring.sum()), mean_float=mean_float.tolist())


def render_tile(sheet, image, x, y, size=240):
    thumb = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    sheet.paste(thumb, (x+(size-thumb.width)//2, y+(size-thumb.height)//2))


def summarize_images(prepared):
    for prompt_kind in ['empty', 'described']:
        sheet = Image.new('RGB', (1320, 1370), '#f2f5f7')
        d = ImageDraw.Draw(sheet)
        d.text((26, 16), '条件图填充对比：纯白 vs 周围平均色', font=font(28), fill='#203949')
        d.text((26, 60), f'{"空提示词" if prompt_kind == "empty" else "描述性提示词"} · Seed 42 · 2 steps · CFG 0 · ControlNet 0.3 · 外侧 24 px 取色', font=font(17), fill='#627481')
        headers = ['原图与遮罩位置', '白色条件图（局部）', '平均色条件图（局部）', '白色填充结果（局部）', '平均色填充结果（局部）']
        for i, title in enumerate(headers):
            d.text((26+i*258, 102), title, font=font(17), fill='#203949')
        for row, case in enumerate(CASES):
            data = prepared[case['id']]
            y = 143+row*299
            d.text((26, y), f'{case["title"]}   周围均色 RGB{data["mean"]}', font=font(18), fill='#203949')
            bbox = crop_box(data['mask'])
            overlay = Image.composite(Image.new('RGB', data['image'].size, '#ed669b'), data['image'], data['mask'].point(lambda v: int(v*.35)))
            pictures = [overlay]
            for mode in ['white', 'mean']:
                control = data['controls'][mode].crop(data['extent']).resize(data['image'].size, Image.Resampling.LANCZOS)
                pictures.append(control.crop(bbox))
            for mode in ['white', 'mean']:
                pictures.append(Image.open(OUT/case['id']/f'{prompt_kind}_seed42_{mode}.png').crop(bbox))
            for col, picture in enumerate(pictures):
                render_tile(sheet, picture, 26+col*258, y+33)
        sheet.save(OUT/f'overview_{prompt_kind}.png')

    for case in CASES:
        data = prepared[case['id']]
        bbox = crop_box(data['mask'])
        sheet = Image.new('RGB', (1050, 1350), '#f2f5f7')
        d = ImageDraw.Draw(sheet)
        d.text((25, 16), f'{case["title"]} — 两个提示词 × 两个种子', font=font(26), fill='#203949')
        d.text((25, 58), '左：白色条件图生成结果    右：周围平均色条件图生成结果（均为局部放大）', font=font(18), fill='#627481')
        for row, (kind, seed) in enumerate((kind, seed) for kind in ['empty','described'] for seed in SEEDS):
            y = 103+row*307
            d.text((25, y), f'{"空提示词" if kind == "empty" else "描述性提示词"} / Seed {seed}', font=font(18), fill='#203949')
            for col, mode in enumerate(['white','mean']):
                im = Image.open(OUT/case['id']/f'{kind}_seed{seed}_{mode}.png')
                render_tile(sheet, im.crop(bbox), 100+col*490, y+30, 260)
        sheet.save(OUT/case['id']/'all_pairs.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = {}
    for case in CASES:
        data = prepare(case)
        prepared[case['id']] = data
        folder = OUT/case['id']
        folder.mkdir(exist_ok=True)
        data['image'].save(folder/'original.png')
        data['mask'].save(folder/'mask.png')
        for mode, image in data['controls'].items():
            image.save(folder/f'control_{mode}.png')
        print(f'Prepared {case["id"]}: mean={data["mean"]}, ring_pixels={data["ring_pixels"]}', flush=True)
    start = time.perf_counter()
    pipe = engine.get_pipeline()
    pipe.set_progress_bar_config(disable=True)
    load_seconds = time.perf_counter()-start
    results, pairs = [], []
    baseline_exact = None
    for case in CASES:
        data = prepared[case['id']]
        original = np.asarray(data['image'])
        inside = np.asarray(data['mask']) > 0
        for kind in ['empty', 'described']:
            prompt = '' if kind == 'empty' else case['prompt']
            for seed in SEEDS:
                paired = {}
                for mode in ['white', 'mean']:
                    started = time.perf_counter()
                    with torch.inference_mode():
                        raw = pipe(prompt=prompt, negative_prompt='low quality, bad anatomy, blurry',
                                   image=data['init'], mask_image=data['work_mask'], control_image=data['controls'][mode],
                                   strength=1.0, num_inference_steps=2, guidance_scale=0,
                                   controlnet_conditioning_scale=SCALE,
                                   generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
                    torch.cuda.synchronize()
                    restored = raw.crop(data['extent']).resize(data['image'].size, Image.Resampling.LANCZOS)
                    result = Image.composite(restored, data['image'], data['mask'])
                    seconds = time.perf_counter()-started
                    arr = np.asarray(result)
                    assert np.array_equal(arr[~inside], original[~inside]), 'Unmasked pixels changed'
                    filename = f'{kind}_seed{seed}_{mode}.png'
                    result.save(OUT/case['id']/filename)
                    paired[mode] = arr
                    results.append(dict(case=case['id'], prompt_kind=kind, prompt=prompt, seed=seed,
                                        fill=mode, seconds=round(seconds,3), file=f'{case["id"]}/{filename}',
                                        outside_mask_unchanged=True))
                    print(f'{len(results):02d}/32 {case["id"]} {kind} seed={seed} {mode} {seconds:.2f}s', flush=True)
                    if baseline_exact is None and mode == 'white':
                        baseline = engine.inpaint_image(data['image'], data['mask'], prompt, seed=seed)
                        baseline_exact = bool(np.array_equal(arr, np.asarray(baseline)))
                        assert baseline_exact, 'Experimental white baseline differs from production engine'
                delta = np.abs(paired['white'].astype(float)-paired['mean'].astype(float))[inside]
                pairs.append(dict(case=case['id'], prompt_kind=kind, seed=seed,
                                  pair_mae_0_255=round(float(delta.mean()),4)))
    manifest = dict(settings=dict(steps=2, strength=1.0, guidance_scale=0, controlnet_scale=SCALE,
                                 seeds=SEEDS, ring_width_working_pixels=RING,
                                 mean_method='arithmetic mean of encoded RGB in square-dilated exterior band; excludes mask and letterbox padding'),
                    versions=dict(torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__),
                    gpu=torch.cuda.get_device_name(), load_seconds=load_seconds, baseline_matches_production=baseline_exact,
                    engine_sha256=hashlib.sha256((ROOT/'sdxl_lightning_inpaint.py').read_bytes()).hexdigest(),
                    cases=[{**case, 'mean_rgb': prepared[case['id']]['mean'], 'ring_pixels': prepared[case['id']]['ring_pixels']} for case in CASES],
                    results=results, pairs=pairs)
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUT/'pairs.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=pairs[0].keys())
        writer.writeheader(); writer.writerows(pairs)
    summarize_images(prepared)
    print('Completed. White baseline matches production; all unmasked pixels preserved.', flush=True)


if __name__ == '__main__':
    main()
