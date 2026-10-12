"""Five unseen project images x five prompt types x three fixed seeds.

Uses the downloaded SD1.5 inpainting checkpoint and established preprocessing.
No application code or shared Python environment is changed.
"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from compare_inpaint_models import (ROOT, MODELS, CONFIG, NEGATIVE, load_model, infer,
                                    prepare, boundary_jump, save_json, crop_box, font,
                                    render_tile, torch, np, diffusers, transformers)
from PIL import Image, ImageDraw

OUT = ROOT / 'experiments' / 'sd15_prompt_evaluation'
SEEDS = [42, 123, 2026]
KINDS = {
    'empty': '空提示词',
    'short_en': '简短英文',
    'short_zh': '对应中文',
    'detailed': '详细英文',
    'edit': '指定新内容',
}
CASES = [
    dict(id='anime', title='动漫眼睛 / 局部结构', source='output_1906.png',
         shape='ellipse', box=(.27,.285,.48,.44), task='补回左侧眼睛；新内容组改为黑色眼罩',
         prompts=dict(empty='', short_en='a blue-gray anime eye.', short_zh='一只蓝灰色的动漫眼睛。',
             detailed='a single blue-gray anime eye, delicate dark eyelashes, pale skin, soft shading, matching the other eye and the surrounding hand-drawn anime portrait, fine clean linework',
             edit='a black leather eyepatch covering one eye, a thin black strap, matching the hand-drawn anime portrait style')),
    dict(id='geometry', title='几何抽象画 / 纹理延续', source='output_350.png',
         shape='rectangle', box=(.36,.36,.66,.64), task='延续几何色块；新内容组加入白色五角星',
         prompts=dict(empty='', short_en='an abstract painting of colorful triangles.', short_zh='彩色三角形组成的抽象画。',
             detailed='an abstract painting of interlocking triangular facets in coral orange, muted blue, teal and gold, crisp angular edges, layered flat shapes, matching the surrounding geometric composition',
             edit='one large white five-pointed star centered among colorful geometric triangles, sharp clean edges, flat graphic painting')),
    dict(id='pagoda', title='古塔线描 / 建筑线条', source='output_4051.png',
         shape='rectangle', box=(.415,.25,.665,.485), task='延续瓦顶与塔身；新内容组加入圆形钟面',
         prompts=dict(empty='', short_en='gray tiled roofs of a traditional Chinese pagoda.', short_zh='中国传统古塔的灰色瓦屋顶。',
             detailed='traditional Chinese pagoda roofs with curved eaves, fine parallel gray roof tiles and timber beams, delicate monochrome ink linework on pale cream paper, continuous architectural perspective, matching the surrounding drawing',
             edit='one large round clock face mounted on the front of the pagoda, visible dark clock hands and roman numerals, delicate monochrome ink drawing on cream paper')),
    dict(id='lineart', title='红色鸟形线稿 / 线条连接', source='333.png',
         shape='rectangle', box=(.35,.25,.62,.50), task='连接红色轮廓线；新内容组改成蓝色羽翼线条',
         prompts=dict(empty='', short_en='a red outline drawing of a bird on a dark gray background.', short_zh='深灰背景上的红色鸟形线稿。',
             detailed='a minimal red outline drawing of a flying bird, thin continuous red strokes connecting the visible wing and body contours, uniform dark gray background, simple flat line art',
             edit='a bright blue outline of a bird wing on a dark gray background, simple thin blue strokes, minimal flat line drawing')),
    dict(id='studio', title='黑白摄影棚 / 移除地面物体', source='output_landscape.png',
         shape='rectangle', box=(.77,.82,.96,.97), task='移除右下角皮包和碎片；新内容组换成红色花瓶',
         prompts=dict(empty='', short_en='an empty gray studio floor.', short_zh='空无一物的灰色摄影棚地面。',
             detailed='an empty gray studio floor with subtle photographic grain, soft shadows consistent with the adjacent floor, continuous clean floor surface, monochrome black and white studio photograph',
             edit='one small bright red ceramic vase standing on the studio floor, realistic smooth glazed pottery, soft contact shadow, studio photograph')),
]


def production_hashes():
    files = [ROOT/'sdxl_lightning_inpaint.py', ROOT/'web_server.py', ROOT/'start_web.ps1', *sorted((ROOT/'web').glob('*'))]
    return {str(p.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def inputs(case):
    original = Image.open(ROOT / case['source']).convert('RGB')
    mask = Image.new('L', original.size)
    box = tuple(round(v*original.size[i%2]) for i,v in enumerate(case['box']))
    draw = ImageDraw.Draw(mask)
    getattr(draw, case['shape'])(box, fill=255)
    return original, mask


def export_previews():
    sheet = Image.new('RGB',(1540,410),'#f2f5f7'); d=ImageDraw.Draw(sheet)
    d.text((22,12),'SD 1.5 提示词实验 · 五张新图片与固定遮罩',font=font(26),fill='#203949')
    for col,case in enumerate(CASES):
        original,mask=inputs(case); folder=OUT/case['id'];folder.mkdir(exist_ok=True)
        original.save(folder/'original.png');mask.save(folder/'mask.png')
        overlay=Image.composite(Image.new('RGB',original.size,'#ed669b'),original,mask.point(lambda p:int(p*.45)))
        overlay.save(folder/'overlay.png')
        canvas,wm,_=prepare(original,mask,512);canvas.save(folder/'input_512.png');wm.save(folder/'mask_512.png')
        bbox=crop_box(mask)
        for name,im in [('original',original),('mask',mask),('overlay',overlay)]:im.crop(bbox).save(folder/f'{name}_crop.png')
        d.text((22+col*305,58),case['title'],font=font(17),fill='#203949')
        render_tile(sheet,overlay,22+col*305,94,280)
        d.text((22+col*305,380),case['source'],font=font(15),fill='#617781')
    sheet.save(OUT/'inputs.png')


def summarize(records):
    summary={'count':len(records),'median_seconds':statistics.median(r['seconds'] for r in records.values()),
             'min_seconds':min(r['seconds'] for r in records.values()),'max_seconds':max(r['seconds'] for r in records.values()),
             'all_outside_unchanged':all(r['outside_mask_unchanged'] for r in records.values()),
             'safety_flag_count':sum(any(r['nsfw_content_detected'] or []) for r in records.values()),'by_prompt':{},'prompt_changes':[]}
    summary['valid_output_count']=len(records)-summary['safety_flag_count']
    for kind in KINDS:
        rows=[r for r in records.values() if r['kind']==kind]
        valid=[r for r in rows if not any(r['nsfw_content_detected'] or [])]
        summary['by_prompt'][kind]={'attempts':len(rows),'valid_outputs':len(valid),
                                   'median_seconds':statistics.median(r['seconds'] for r in rows)}
    for case in CASES:
        _,mask=inputs(case);inside=np.asarray(mask)>0
        for seed in SEEDS:
            baseline=records[f'{case["id"]}/empty/{seed}']
            empty=np.asarray(Image.open(OUT/baseline['file'])).astype(float)
            for kind in list(KINDS)[1:]:
                r=records[f'{case["id"]}/{kind}/{seed}']
                arr=np.asarray(Image.open(OUT/r['file'])).astype(float)
                eligible=not any(baseline['nsfw_content_detected'] or []) and not any(r['nsfw_content_detected'] or [])
                summary['prompt_changes'].append(dict(case=case['id'],kind=kind,seed=seed,
                    masked_rgb_mae_vs_empty=round(float(np.abs(arr-empty)[inside].mean()),4) if eligible else None,
                    excluded_due_to_safety_replacement=not eligible))
    save_json(OUT/'summary.json',summary)
    return summary


def mark_replaced(draw,x,y):
    draw.rectangle((x,y,x+257,y+30),fill='#fff0d6')
    draw.text((x+7,y+4),'检测器置黑 · 非有效补全',font=font(17),fill='#845d2e')


def render(records):
    for seed in SEEDS:
        sheet=Image.new('RGB',(1700,1710),'#f2f5f7');d=ImageDraw.Draw(sheet)
        d.text((24,14),f'SD 1.5 Inpainting · 提示词对照 · Seed {seed}',font=font(29),fill='#203949')
        d.text((24,59),'512 × 512 · 30 steps · CFG 7.5 · 相同遮罩 · 硬合成无羽化 · 中文对应简短英文',font=font(18),fill='#617781')
        for col,title in enumerate(['原图与遮罩']+list(KINDS.values())):d.text((24+col*278,101),title,font=font(19),fill='#203949')
        for row,case in enumerate(CASES):
            original,mask=inputs(case);box=crop_box(mask);y=141+row*310
            d.text((24,y),case['title'],font=font(19),fill='#203949')
            pics=[Image.open(OUT/case['id']/'overlay.png')]+[Image.open(OUT/records[f'{case["id"]}/{kind}/{seed}']['file']) for kind in KINDS]
            for col,im in enumerate(pics):render_tile(sheet,im.crop(box),24+col*278,y+33,258)
            for col,kind in enumerate(KINDS,1):
                if any(records[f'{case["id"]}/{kind}/{seed}']['nsfw_content_detected'] or []):mark_replaced(d,24+col*278,y+33)
        sheet.save(OUT/f'overview_seed{seed}.png')
    for case in CASES:
        _,mask=inputs(case);box=crop_box(mask)
        sheet=Image.new('RGB',(1700,1120),'#f2f5f7');d=ImageDraw.Draw(sheet)
        d.text((24,14),case['title']+' · 全部提示词与种子',font=font(28),fill='#203949')
        for col,title in enumerate(['原图（局部）']+list(KINDS.values())):d.text((24+col*278,66),title,font=font(19),fill='#203949')
        for row,seed in enumerate(SEEDS):
            y=106+row*330;d.text((24,y),f'Seed {seed}',font=font(19),fill='#203949')
            pics=[Image.open(OUT/case['id']/'original.png')]+[Image.open(OUT/records[f'{case["id"]}/{kind}/{seed}']['file']) for kind in KINDS]
            for col,im in enumerate(pics):render_tile(sheet,im.crop(box),24+col*278,y+32,258)
            for col,kind in enumerate(KINDS,1):
                if any(records[f'{case["id"]}/{kind}/{seed}']['nsfw_content_detected'] or []):mark_replaced(d,24+col*278,y+32)
        sheet.save(OUT/case['id']/'all_prompts.png')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);export_previews()
    if args.prepare_only:return
    record_path=OUT/'results.json';records=json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else {}
    if args.render_only:summarize(records);render(records);return
    before=production_hashes()
    missing=[(c,k,s) for c in CASES for k in KINDS for s in SEEDS if f'{c["id"]}/{k}/{s}' not in records]
    manifest_path=OUT/'manifest.json'
    if missing:
        started=time.perf_counter();model=load_model('sd15_inpaint');torch.cuda.synchronize();load_seconds=time.perf_counter()-started
        tokenization={}
        for case in CASES:
            for kind,prompt in case['prompts'].items():
                count=len(model.tokenizer(prompt,truncation=False)['input_ids']);tokenization[f'{case["id"]}/{kind}']={'tokens':count,'limit':model.tokenizer.model_max_length,'truncated':count>model.tokenizer.model_max_length}
        assert not any(v['truncated'] for v in tokenization.values()), 'Shorten prompts before comparing truncation-confounded text'
        manifest=dict(model='stable-diffusion-v1-5/stable-diffusion-inpainting',revision='8a4288a76071f7280aedbdb3253bdb9e9d5d84bb',
            cases=CASES,kinds=KINDS,seeds=SEEDS,config=CONFIG['sd15_inpaint'],negative_prompt=NEGATIVE,scheduler=dict(model.scheduler.config),
            gpu=torch.cuda.get_device_name(),versions={'torch':torch.__version__,'diffusers':diffusers.__version__,'transformers':transformers.__version__},
            tokenization=tokenization,load_seconds=round(load_seconds,3),production_before=before,
            source_sha256={c['source']:hashlib.sha256((ROOT/c['source']).read_bytes()).hexdigest() for c in CASES},
            notes=['Project images, all different from the prior four-case benchmark; no external dataset.',
                   'Same mask and random noise for each seed within a case. Only positive prompt changes.',
                   'Short English and Chinese are paired translations. Detailed and edit prompts are distinct conditions.',
                   'White masked input, 512 square fit with gray padding. Hard composite at original size, no feathering.',
                   'Timings exclude loading, three warmups, preprocessing and PNG save; include inference, sync, restore/composite.',
                   'Boundary jump and difference from empty prompt are not semantic quality or instruction adherence scores.'])
        save_json(manifest_path,manifest)
        original,mask=inputs(CASES[0]);canvas,wm,_=prepare(original,mask,512)
        with torch.inference_mode():
            for _ in range(3):infer(model,'sd15_inpaint',canvas,wm,'',0)
        for case,kind,seed in missing:
            original,mask=inputs(case);canvas,wm,extent=prepare(original,mask,512)
            torch.cuda.synchronize();started=time.perf_counter()
            with torch.inference_mode():raw,safety=infer(model,'sd15_inpaint',canvas,wm,case['prompts'][kind],seed)
            torch.cuda.synchronize();restored=raw.crop(extent).resize(original.size,Image.Resampling.LANCZOS)
            result=Image.composite(restored,original,mask);seconds=time.perf_counter()-started
            outside=np.asarray(mask)==0;assert np.array_equal(np.asarray(result)[outside],np.asarray(original)[outside])
            filename=f'{case["id"]}/{kind}_seed{seed}.png';result.save(OUT/filename);raw.save((OUT/filename).with_stem(Path(filename).stem+'_raw'))
            result.crop(crop_box(mask)).save((OUT/filename).with_stem(Path(filename).stem+'_crop'))
            key=f'{case["id"]}/{kind}/{seed}';records[key]=dict(case=case['id'],kind=kind,seed=seed,prompt=case['prompts'][kind],
                seconds=round(seconds,3),file=filename,boundary_jump=boundary_jump(result,mask),original_boundary_jump=boundary_jump(original,mask),
                outside_mask_unchanged=True,nsfw_content_detected=safety)
            save_json(record_path,records);print(f'{len(records)}/75 {key} {seconds:.3f}s',flush=True)
        del model;torch.cuda.empty_cache()
    after=production_hashes();assert before==after
    save_json(OUT/'production_unchanged.json',{'all_unchanged':True,'sha256':after})
    assert len(records)==75
    summary=summarize(records);render(records)
    print(json.dumps({k:v for k,v in summary.items() if k!='prompt_changes'},indent=2),flush=True)


if __name__=='__main__':main()
