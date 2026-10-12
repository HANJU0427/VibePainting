"""Diagnose SDXL <1 strength sensitivity to the hidden-region initialization."""
import argparse
import json
import time

from compare_inpaint_models import (CASES, CONFIG, OUT, ROOT, PREVIOUS, source, prepare,
                                    load_model, infer, boundary_jump, save_json, crop_box, font, render_tile)
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw

DEST = OUT / 'sdxl_initialization'
MODES = ['original', 'mean', 'white_strength1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-original', action='store_true')
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    p = DEST / 'results.json'
    records = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    model = load_model('sdxl_inpaint')
    original, mask = source(CASES[0])
    canvas, wm, _ = prepare(original, mask, 1024)
    with torch.inference_mode():
        for _ in range(3):
            infer(model, 'sdxl_inpaint', canvas, wm, '', 0)
    for mode in MODES:
        combinations = [(kind, seed) for kind in ['empty', 'described'] for seed in [42, 123]] if args.full_original and mode == 'original' else [('described', 42)]
        for case in CASES:
            original, mask = source(case)
            canvas, wm, extent = prepare(original, mask, 1024)
            x, y, x1, y1 = extent
            intact = Image.new('RGB', (1024, 1024), (127,127,127))
            intact.paste(original.resize((x1-x,y1-y), Image.Resampling.LANCZOS), (x,y))
            if mode == 'original':
                canvas = intact
            elif mode == 'mean':
                m = np.asarray(wm) > 0
                valid = np.zeros((1024,1024), dtype=bool); valid[y:y1,x:x1] = True
                ring = (np.asarray(wm.filter(ImageFilter.MaxFilter(49))) > 0) & ~m & valid
                color = tuple(np.rint(np.asarray(intact)[ring].mean(axis=0)).astype(int).tolist())
                canvas = Image.composite(Image.new('RGB', (1024,1024), color), intact, wm)
            CONFIG['sdxl_inpaint']['strength'] = 1.0 if mode == 'white_strength1' else .99
            for kind, seed in combinations:
                key = f'{mode}/{case["id"]}/{kind}/{seed}'
                if key in records:
                    continue
                prompt = case['prompt'] if kind == 'described' else ''
                torch.cuda.synchronize(); started = time.perf_counter()
                with torch.inference_mode():
                    raw, _ = infer(model, 'sdxl_inpaint', canvas, wm, prompt, seed)
                torch.cuda.synchronize()
                restored = raw.crop(extent).resize(original.size, Image.Resampling.LANCZOS)
                result = Image.composite(restored, original, mask)
                seconds = time.perf_counter()-started
                outside = np.asarray(mask)==0
                assert np.array_equal(np.asarray(result)[outside], np.asarray(original)[outside])
                filename = f'{case["id"]}_{mode}_{kind}_seed{seed}.png'
                result.save(DEST / filename)
                raw.save(DEST / filename.replace('.png','_raw.png'))
                records[key] = dict(case=case['id'], mode=mode, prompt_kind=kind, seed=seed,
                                    strength=CONFIG['sdxl_inpaint']['strength'], seconds=round(seconds,3),
                                    file='sdxl_initialization/'+filename,
                                    boundary_jump=boundary_jump(result,mask), outside_mask_unchanged=True)
                save_json(p, records)
                print(key, f'{seconds:.3f}s', records[key]['boundary_jump'], flush=True)
    sheet = Image.new('RGB', (1420,1500), '#f2f5f7'); d=ImageDraw.Draw(sheet)
    d.text((24,15),'SDXL 补充实验 · 遮罩内初始图像与重绘强度',font=font(28),fill='#203949')
    d.text((24,62),'相同描述性提示词 · Seed 42 · 20 steps · CFG 8 · 原始内容组会利用遮罩内已有像素',font=font(18),fill='#627481')
    for col,title in enumerate(['原图','预填白色 / 0.99','保留原始内容 / 0.99','周围平均色 / 0.99','预填白色 / 1.0']):
        d.text((24+278*col,106),title,font=font(18),fill='#203949')
    for row,case in enumerate(CASES):
        original,mask=source(case); y=150+327*row; box=crop_box(mask)
        d.text((24,y),case['title'],font=font(20),fill='#203949')
        images=[original,Image.open(OUT/case['id']/'sdxl_inpaint_described_seed42.png')]
        images += [Image.open(OUT/records[f'{mode}/{case["id"]}/described/42']['file']) for mode in MODES]
        for col,im in enumerate(images): render_tile(sheet,im.crop(box),24+278*col,y+37,258)
    sheet.save(DEST/'overview.png')


if __name__=='__main__':
    main()
