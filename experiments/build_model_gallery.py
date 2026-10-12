"""Build a self-contained, read-only results viewer (separate from the app UI)."""
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'model_comparison'


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    records = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    extra = json.loads((OUT / 'sdxl_initialization' / 'results.json').read_text(encoding='utf-8'))
    cases = manifest['cases']
    for case in cases:
        folder = OUT / case['id']
        mask = Image.open(folder / 'mask.png')
        x0, y0, x1, y1 = mask.getbbox()
        size = min(int(max(x1-x0, y1-y0)*1.5), mask.width, mask.height)
        x = max(0, min(mask.width-size, (x0+x1)//2-size//2))
        y = max(0, min(mask.height-size, (y0+y1)//2-size//2))
        case['crop_box'] = [x, y, x+size, y+size]
        paths = [folder / 'original.png'] + [OUT / r['file'] for r in records.values() if r['case'] == case['id']]
        paths += [OUT / r['file'] for r in extra.values() if r['case'] == case['id']]
        for path in paths:
            Image.open(path).crop(case['crop_box']).save(path.with_stem(path.stem + '_crop'))
    data = json.dumps({'cases': cases, 'records': records, 'extra': extra, 'config': manifest['config']}, ensure_ascii=False)
    html = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VibePainting · 补全模型实验</title>
<style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#193443;background:#f4f6f7;font-synthesis:none}*{box-sizing:border-box}body{margin:0;padding:32px;max-width:2000px;margin:auto}header{display:flex;justify-content:space-between;gap:24px;align-items:center;border-bottom:1px solid #cbd8dd;padding-bottom:22px}h1{font-size:28px;letter-spacing:-.5px;margin:8px 0}p{line-height:1.7;margin:10px 0;color:#5a6c75}.eyebrow{font-size:12px;letter-spacing:2px;color:#187e75}.badge{background:#e1eeea;color:#226658;padding:8px 14px;border-radius:5px;font-size:13px;white-space:nowrap}.controls{display:flex;gap:22px;flex-wrap:wrap;padding:24px 0;align-items:end}label{display:grid;gap:7px;font-size:13px;color:#52636b}select{font:inherit;font-size:15px;color:#193443;background:white;border:1px solid #b7c9d0;border-radius:5px;padding:10px 32px 10px 12px;min-height:42px}#grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px}.card{background:white;border:1px solid #d8e0e4;border-radius:7px;overflow:hidden}.card h2{font-size:16px;margin:0;padding:17px 14px 12px}.card img{width:100%;aspect-ratio:1;object-fit:contain;background:#e8edef;display:block}.card a{display:block}.meta{font-size:13px;color:#5a6c75;padding:12px 14px;min-height:80px;line-height:1.8}#prompt{font-size:14px;background:#e9efef;padding:14px 18px;border-radius:5px;word-break:break-word}.note{font-size:13px;line-height:1.9;margin-top:22px;max-width:1120px}footer{border-top:1px solid #cbd8dd;padding-top:15px;margin-top:30px;font-size:13px;display:flex;gap:24px}footer a{color:#197467}select:focus-visible,a:focus-visible{outline:3px solid #ecb85d;outline-offset:2px}@media(max-width:1100px){#grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:700px){body{padding:18px}header{display:block}#grid{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:23px}.controls{gap:12px}}
</style>
<header><div><div class="eyebrow">VIBEPAINTING / LOCAL EXPERIMENT</div><h1>补全模型：边缘能否接得更自然</h1><p>相同图片与遮罩，比较当前 Lightning、SD 1.5 Inpainting、SDXL Inpainting 和 LaMa。</p></div><span class="badge">独立实验 · 不影响应用前端</span></header>
<div class="controls"><label>示例<select id="case"></select></label><label>提示词<select id="kind"><option value="empty">留空</option><option value="described">描述性提示词</option></select></label><label>随机种子<select id="seed"><option>42</option><option>123</option></select></label><label>查看范围<select id="view"><option value="crop">遮罩周围放大</option><option value="full">完整图片</option></select></label><label>SDXL 遮罩内初始内容<select id="sdxl"><option value="white">预填白色 · 强度 0.99</option><option value="original">保留原图 · 强度 0.99</option><option value="mean">周围平均色 · 强度 0.99（补充）</option><option value="white_strength1">预填白色 · 强度 1.0（补充）</option></select></label></div>
<p id="prompt"></p><main id="grid"></main>
<p class="note">所有结果均使用硬遮罩合成，没有羽化；遮罩外的像素与原图完全相同。点击图片可打开原分辨率文件。LaMa 无提示词和随机种子，同一结果会在各组中重复展示。SD 1.5 在 512 × 512 工作，其余在 1024 × 1024 工作，均保持原图比例。<br>SDXL 的“保留原图”组会利用遮罩内原有内容，对应现有图片的编辑场景；“周围平均色”和“强度 1.0”是追加诊断，仅测试描述性提示词、Seed 42，选择后会自动切换到该组。这里改变的是初始图像，不是 ControlNet 条件图。<br>“边界色差”是跨遮罩边界的相邻像素 RGB 平均差（0–255），仅辅助观察接缝；模糊、真实物体边缘及不同分辨率都会影响它，不能当作画质排名。耗时不含模型加载、预热和图片保存；当前模型的结果与耗时复用上一轮实验。</p>
<footer><a href="REPORT.md">实验记录</a><a href="overview_empty.png">空提示词总览</a><a href="overview_described.png">描述性提示词总览</a><a href="sdxl_initialization/overview.png">SDXL 输入对照</a><a href="manifest.json">参数与环境</a><a href="results.json">逐张数据</a></footer>
<script>
const data=__DATA__;
const byId=id=>document.getElementById(id);
for(const c of data.cases){const o=document.createElement('option');o.value=c.id;o.textContent=c.title;byId('case').append(o)}
function show(){
const mode=byId('sdxl').value,pilot=['mean','white_strength1'].includes(mode);byId('kind').disabled=pilot;byId('seed').disabled=pilot;if(pilot){byId('kind').value='described';byId('seed').value='42'}
const c=data.cases.find(c=>c.id===byId('case').value),kind=byId('kind').value,seed=Number(byId('seed').value),crop=byId('view').value==='crop';
byId('prompt').textContent=kind==='empty'?'正向提示词为空；扩散模型沿用相同的负向提示词。':'正向提示词：'+c.prompt;
byId('grid').replaceChildren();
for(const name of ['original',...Object.keys(data.config)]){
const key=name==='lama'?`lama/${c.id}/none/null`:`${name}/${c.id}/${kind}/${seed}`;
// Python serializes the LaMa key's seed as None; JSON record seed itself is null.
let r=name==='lama'?Object.values(data.records).find(r=>r.model==='lama'&&r.case===c.id):data.records[key];
if(name==='sdxl_inpaint'&&mode!=='white')r=data.extra[`${mode}/${c.id}/${kind}/${seed}`];
const file=name==='original'?`${c.id}/original.png`:r.file,src=crop?file.replace('.png','_crop.png'):file;
const card=document.createElement('article');card.className='card';const title=document.createElement('h2');title.textContent=name==='original'?'原图':data.config[name].label;
const link=document.createElement('a');link.href=file;link.target='_blank';link.rel='noopener';const im=document.createElement('img');im.src=src;im.alt=c.title+' — '+title.textContent;link.append(im);
const meta=document.createElement('div');meta.className='meta';meta.textContent=name==='original'?'用于观察原有颜色、纹理和轮廓':`${r.seconds.toFixed(3)} 秒 · 边界色差 ${r.boundary_jump.toFixed(2)}`;
if(name!=='original'){const line=document.createElement('div');line.textContent=name==='lama'?'1024 · 无提示词':`${data.config[name].size} · ${data.config[name].steps} 步 · CFG ${data.config[name].cfg}`;meta.append(line)}
card.append(title,link,meta);byId('grid').append(card);
}}
for(const id of ['case','kind','seed','view','sdxl'])byId(id).addEventListener('change',show);show();
</script></html>'''.replace('__DATA__', data.replace('</', '<\\/'))
    (OUT / 'index.html').write_text(html, encoding='utf-8')
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
