"""Read-only experiment gallery, separate from the application's frontend."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/'sd15_prompt_evaluation'


def main():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    records=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    notes_path=OUT/'observations.json'
    notes=json.loads(notes_path.read_text(encoding='utf-8')) if notes_path.exists() else {}
    data=json.dumps(dict(manifest=manifest,records=records,summary=summary,notes=notes),ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SD 1.5 · 不同图片与提示词评估</title>
<style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#193443;background:#f4f6f7;font-synthesis:none}*{box-sizing:border-box}body{margin:0;padding:30px;max-width:2100px;margin:auto}.eyebrow{color:#197f77;font-size:12px;letter-spacing:2px}h1{font-size:28px;margin:10px 0}p{color:#596e79;line-height:1.75}header{border-bottom:1px solid #cbd8dd;padding-bottom:15px}.stats{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}.stats span{font-size:13px;color:#31675f;background:#e3efeb;padding:7px 11px;border-radius:4px}.controls{display:flex;gap:20px;align-items:end;flex-wrap:wrap;padding:10px 0 22px}label{font-size:13px;color:#526873;display:grid;gap:7px}select{font:inherit;font-size:15px;color:#193443;background:white;border:1px solid #b7c9d0;border-radius:5px;padding:10px 30px 10px 12px;min-height:42px}.check{display:flex;align-items:center;gap:7px;min-height:42px;cursor:pointer}.check input{accent-color:#197f77;width:17px;height:17px}#task{font-size:14px;margin:0 0 20px}#grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;align-items:stretch}.card{min-width:0;background:white;border:1px solid #d8e0e4;border-radius:7px;overflow:hidden;display:flex;flex-direction:column}.card h2{font-size:16px;margin:0;padding:15px 12px}.card img{width:100%;aspect-ratio:1;object-fit:contain;background:#e8edef;display:block}.image{display:block}.meta{font-size:12px;color:#617781;padding:11px 12px 0}.prompt{font-size:13px;color:#344f5d;line-height:1.65;padding:0 12px;overflow-wrap:anywhere}.raw{font-size:12px;margin:auto 12px 14px;padding-top:10px;color:#1e766d}.muted{color:#85939a}#note{margin-top:23px;padding:15px 18px;border-left:3px solid #699c8c;background:#edf2f0;font-size:14px;line-height:1.8}.method{font-size:13px;line-height:1.85;max-width:1300px;margin-top:20px;color:#617781}details{margin-top:9px}summary{cursor:pointer;color:#286f66}footer{display:flex;gap:20px;flex-wrap:wrap;border-top:1px solid #cbd8dd;padding-top:18px;margin-top:24px;font-size:13px}footer a{color:#1e766d}select:focus-visible,a:focus-visible,input:focus-visible{outline:3px solid #e6b660;outline-offset:2px}@media(max-width:1200px){#grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:700px){body{padding:18px}h1{font-size:23px}.controls{gap:12px}#grid{grid-template-columns:repeat(2,minmax(0,1fr))}.card h2{font-size:14px}}@media(max-width:420px){#grid{grid-template-columns:1fr}}
</style>
<header><div class="eyebrow">VIBEPAINTING / PROMPT EXPERIMENT</div><h1>SD 1.5：换图片，换提示词</h1><p>五张新图片、五种提示词、三个随机种子。观察内容是否听从描述，也观察颜色、轮廓与纹理是否接得上。</p></header>
<div class="stats"><span>75 次生成 · 72 张有效输出 · 3 次检测器置黑</span><span>512 × 512 · 30 步 · CFG 7.5</span><span>固定遮罩 · 无羽化</span><span id="time"></span></div>
<div class="controls"><label>测试场景<select id="case"></select></label><label>随机种子<select id="seed"><option>42</option><option>123</option><option>2026</option></select></label><label>查看范围<select id="view"><option value="crop">遮罩周围放大</option><option value="full">完整图片</option></select></label><label class="check"><input type="checkbox" id="overlay" checked>原图显示遮罩</label></div>
<p id="task"></p><main id="grid"></main><div id="note"></div>
<div class="method">简短英文与对应中文表达相同内容；详细英文增加材质、风格及周围结构描述；“指定新内容”用于测试主动编辑能力，目标与修复原样不同。相同种子的一行中，仅正向提示词改变。原图显示的粉色区域就是固定遮罩，生成结果遮罩外的像素完全不变。<br>所有输入来自项目现有素材，五张均未用于上一轮四场景对比。330 像素的线稿上采样至 512，其余从 1024 缩至 512；结果恢复原图尺寸。这里是小规模实验，不代表所有图片上的表现。<details><summary>测量和提示词设置</summary><p>重绘强度 1.0，DDIM 调度器，负向提示词统一为 <code>low quality, bad anatomy, blurry</code>。“空提示词”指正向留空。所有正向文本均在 CLIP 长度限制内，没有截断。时间不含模型加载、三次预热、输入预处理及 PNG 保存；包括推理与输出合成。边界色差是跨遮罩边界相邻像素的 RGB 平均绝对差（0–255），只辅助观察接缝，不是画质或指令遵循分数。</p></details></div>
<footer><a href="REPORT.md">实验报告与结论</a><a id="overview" href="overview_seed42.png">当前种子总览</a><a id="all" href="anime/all_prompts.png">当前场景全部种子</a><a href="inputs.png">五张原图与遮罩</a><a href="manifest.json">提示词与完整参数</a><a href="results.json">逐张数据</a><a href="http://127.0.0.1:8770/">上一轮模型对比</a></footer>
<script>
const data=__DATA__,el=id=>document.getElementById(id);
for(const c of data.manifest.cases){const opt=document.createElement('option');opt.value=c.id;opt.textContent=c.title;el('case').append(opt)}
el('time').textContent='本机中位耗时 '+data.summary.median_seconds.toFixed(2)+' 秒';
function show(){
const c=data.manifest.cases.find(c=>c.id===el('case').value),seed=Number(el('seed').value),crop=el('view').value==='crop';
el('task').textContent='目标：'+c.task+'　｜　来源：'+c.source;el('grid').replaceChildren();
for(const kind of ['original',...Object.keys(data.manifest.kinds)]){
const r=data.records[`${c.id}/${kind}/${seed}`];const path=kind==='original'?`${c.id}/${el('overlay').checked?'overlay':'original'}.png`:r.file;
const replaced=r&&(r.nsfw_content_detected||[]).some(Boolean);
const card=document.createElement('article');card.className='card';const title=document.createElement('h2');title.textContent=kind==='original'?'原图 / 固定遮罩':data.manifest.kinds[kind];
const a=document.createElement('a');a.className='image';a.href=path;a.target='_blank';a.rel='noopener';const im=document.createElement('img');im.src=crop?path.replace('.png','_crop.png'):path;im.alt=c.title+' — '+title.textContent;a.append(im);
const meta=document.createElement('div');meta.className='meta';meta.textContent=kind==='original'?'供观察原来的纹理、颜色与轮廓':replaced?'模型检测器已置黑；不计入有效补全比较':`${r.seconds.toFixed(3)} 秒 · 边界色差 ${r.boundary_jump.toFixed(2)}`;if(replaced){meta.style.background='#fff0d6';meta.style.color='#845d2e';meta.style.paddingBottom='10px'}
const prompt=document.createElement('p');prompt.className='prompt';prompt.textContent=kind==='original'?c.task:kind==='empty'?'（正向提示词留空）':r.prompt;
card.append(title,a,meta,prompt);
if(kind!=='original'){const raw=document.createElement('a');raw.className='raw';raw.href=r.file.replace('.png','_raw.png');raw.target='_blank';raw.rel='noopener';raw.textContent='查看模型原始 512 输出 ↗';card.append(raw)}
el('grid').append(card)}
el('note').textContent=data.notes[c.id]||'同一个种子横向比较提示词；切换种子，检查观察到的效果是否稳定。';
el('overview').href=`overview_seed${seed}.png`;el('all').href=`${c.id}/all_prompts.png`;
}
for(const id of ['case','seed','view','overlay'])el(id).addEventListener('change',show);show();
</script></html>'''.replace('__DATA__',data)
    (OUT/'index.html').write_text(html,encoding='utf-8');print(OUT/'index.html')


if __name__=='__main__':main()
