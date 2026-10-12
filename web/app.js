'use strict';
const $ = id => document.getElementById(id);
const base = $('image-canvas'), mask = $('mask-canvas');
const baseCtx = base.getContext('2d'), maskCtx = mask.getContext('2d', {willReadFrequently: true});
// Keep hand-painted pixels separate so the restore brush reveals the loaded image.
// The visible base canvas is always the flattened image submitted for generation.
const original = document.createElement('canvas'), paint = document.createElement('canvas');
const originalCtx = original.getContext('2d'), paintCtx = paint.getContext('2d');
let loaded = false, busy = false, restoring = false, drawing = false, mode = 'brush';
let editLayer = 'mask', lastPoint = null, pointerId = null, history = [], future = [], hasMask = false;
let resultUrl = null, resultSource = null, loadVersion = 0;
let modelCatalog = [], selectedModel = null;
const modelSettings = new Map();
const parameterFields = {strength: 'strength', num_inference_steps: 'steps', guidance_scale: 'guidance',
  controlnet_conditioning_scale: 'control', seed: 'seed', negative_prompt: 'negative'};

function modelSpec() { return modelCatalog.find(model => model.id === selectedModel); }
function rememberModelSettings() {
  if (!selectedModel) return;
  modelSettings.set(selectedModel, Object.fromEntries(Object.values(parameterFields).map(id => [id, $(id).value])));
}
function selectModel(id) {
  rememberModelSettings();
  selectedModel = id;
  const spec = modelSpec();
  $('model').value = id;
  const saved = modelSettings.get(id);
  for (const [name, field] of Object.entries(parameterFields)) {
    if (saved) $(field).value = saved[field];
    else if (name in spec.defaults) $(field).value = spec.defaults[name];
    else if (name === 'seed') $(field).value = '';
    else if (name === 'negative_prompt') $(field).value = 'low quality, bad anatomy, blurry';
    $(field).closest('label').hidden = !spec.parameters.includes(name);
  }
  $('model-description').textContent = spec.description;
  $('parameter-hint').textContent = spec.parameter_hint;
  $('generation-parameters').hidden = !spec.parameters.length;
  $('prompt-hint').textContent = spec.supports_prompt
    ? '可留空，让模型根据画面尝试补全；需要指定内容时，建议用英文描述。'
    : 'LaMa 根据周围画面自动修补，不使用提示词；已输入的描述会保留。';
  try { localStorage.setItem('vibepainting-model', id); } catch (_) { /* Storage may be disabled. */ }
  updateControls();
}
$('model').addEventListener('change', () => {
  selectModel($('model').value);
  status(`已选择 ${modelSpec().label}。图片与遮罩已保留，点击生成时生效。`);
});

function status(text, kind = '') {
  $('status').textContent = text;
  $('status-bar').className = `status-bar ${kind}`;
}
function updateControls() {
  const locked = busy || restoring || drawing;
  const spec = modelSpec();
  ['brush', 'eraser', 'eyedropper', 'brush-size', 'paint-color', 'edit-image', 'edit-mask', 'clear', 'save-mask', 'save-image'].forEach(id => $(id).disabled = !loaded || locked);
  document.querySelectorAll('.swatch').forEach(button => button.disabled = !loaded || locked);
  $('show-mask').disabled = !loaded || locked || editLayer === 'image';
  $('undo').disabled = !history.length || locked;
  $('redo').disabled = !future.length || locked;
  $('generate').disabled = !loaded || !hasMask || locked || !spec?.available;
  $('file-input').disabled = locked;
  document.querySelector('.upload-label').classList.toggle('disabled', locked);
  $('use-result').disabled = !resultUrl || locked;
  $('download').disabled = !resultUrl;
  $('compare').disabled = !resultUrl || !resultSource;
  $('generate-form').querySelectorAll('input, textarea').forEach(el => el.disabled = locked);
  $('model').disabled = locked || !modelCatalog.length;
  $('prompt').disabled = locked || !spec?.supports_prompt;
  for (const [name, id] of Object.entries(parameterFields)) $(id).disabled = locked || !spec?.parameters.includes(name);
  $('generate-label').textContent = busy ? '正在生成…' : '生成补全';
}
function setMode(value) {
  if (drawing || busy || restoring || (value === 'eyedropper' && editLayer !== 'image')) return;
  mode = value;
  ['brush', 'eraser', 'eyedropper'].forEach(id => {
    $(id).classList.toggle('active', id === mode);
    $(id).setAttribute('aria-pressed', String(id === mode));
  });
  mask.classList.toggle('sampling', mode === 'eyedropper');
  $('cursor').hidden = true;
}
function syncMaskVisibility() {
  mask.classList.toggle('mask-hidden', editLayer === 'image' || !$('show-mask').checked);
}
function setEditLayer(value) {
  if (drawing || busy || restoring) return;
  editLayer = value;
  const painting = value === 'image';
  ['image', 'mask'].forEach(layer => {
    $(`edit-${layer}`).classList.toggle('active', value === layer);
    $(`edit-${layer}`).setAttribute('aria-pressed', String(value === layer));
  });
  $('paint-colors').hidden = !painting;
  $('eyedropper').hidden = !painting;
  $('eraser-label').textContent = painting ? '还原笔' : '橡皮';
  $('eraser').title = painting ? '擦除手绘笔迹，露出本次载入的图片（E）' : '擦除遮罩（E）';
  $('clear').textContent = painting ? '清除笔迹' : '清空遮罩';
  $('edit-hint').textContent = painting ? '直接修改图片，作为补全输入' : '标记需要 AI 重绘的区域';
  $('canvas-help').textContent = painting
    ? '笔迹会写入补全输入。还原笔只擦除手绘笔迹；绘图时暂时隐藏遮罩。'
    : '粉色区域将被重新生成；遮罩之外保留当前画面。';
  mask.setAttribute('aria-label', painting ? '直接在图像上绘制，按 B 使用画笔，E 使用还原笔，I 吸取颜色' : '用鼠标绘制遮罩，按 B 切换画笔，按 E 切换橡皮');
  if (!painting && mode === 'eyedropper') setMode('brush');
  syncMaskVisibility();
  $('cursor').hidden = true;
  updateControls();
}
function setPaintColor(color) {
  $('paint-color').value = color;
  $('color-value').textContent = color.toUpperCase();
  document.querySelectorAll('.swatch').forEach(button => {
    const active = button.dataset.color === color.toLowerCase();
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}
function renderImage() {
  baseCtx.clearRect(0, 0, base.width, base.height);
  baseCtx.drawImage(original, 0, 0);
  baseCtx.drawImage(paint, 0, 0);
}
function layoutCanvas() {
  if (!loaded) return;
  const available = $('stage').clientWidth - (innerWidth <= 760 ? 32 : 48);
  const width = Math.min(available, 620 * base.width / base.height);
  $('canvas-stack').style.width = `${width}px`;
  $('canvas-stack').style.height = `${width * base.height / base.width}px`;
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('无法读取图片，请使用 PNG、JPEG 或 WebP。'));
    image.src = url;
  });
}
function installImage(image, name) {
  base.width = mask.width = original.width = paint.width = image.naturalWidth;
  base.height = mask.height = original.height = paint.height = image.naturalHeight;
  originalCtx.fillStyle = '#ffffff';
  originalCtx.fillRect(0, 0, original.width, original.height);
  originalCtx.drawImage(image, 0, 0);
  renderImage();
  history = []; future = []; hasMask = false; loaded = true;
  $('empty').hidden = true; $('canvas-stack').hidden = false;
  $('show-mask').checked = true; syncMaskVisibility();
  $('image-info').textContent = `${name} · ${base.width} × ${base.height}`;
  $('image-info').title = $('image-info').textContent;
  layoutCanvas(); updateControls();
  status('可切换「绘制图像」直接涂色，再用「补全遮罩」标记需要生成的区域。');
}
async function openFile(file) {
  if (!file || busy || restoring || drawing) return;
  const version = ++loadVersion;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) return status('请选择 PNG、JPEG 或 WebP 图片。', 'error');
  if (file.size > 20 * 1024 * 1024) return status('图片文件超过 20 MB，请压缩后重试。', 'error');
  const url = URL.createObjectURL(file);
  try {
    const image = await loadImage(url);
    if (version !== loadVersion || busy) return;
    if (image.naturalWidth * image.naturalHeight > 16000000) throw new Error('图片超过 1600 万像素，请缩小后重试。');
    resultUrl = resultSource = null;
    $('result-image').hidden = true; $('result-image').removeAttribute('src');
    $('result-empty').hidden = false; $('result-meta').textContent = '等待生成'; $('result-seed').textContent = '';
    installImage(image, file.name);
  } catch (error) { status(error.message, 'error'); }
  finally { URL.revokeObjectURL(url); $('file-input').value = ''; }
}
function snapshot(layer = editLayer) {
  return {layer, data: (layer === 'image' ? paint : mask).toDataURL('image/png')};
}
function trimHistory() {
  // Sparse paint/mask PNGs are cheaper than copying the full photograph per stroke.
  let bytes = history.reduce((total, entry) => total + entry.data.length * 2, 0);
  while (history.length > 1 && (history.length > 20 || bytes > 64 * 1024 * 1024)) {
    bytes -= history.shift().data.length * 2;
  }
}
function remember() {
  history.push(snapshot());
  future = [];
  trimHistory();
}
async function restoreHistory(redo = false) {
  const from = redo ? future : history, to = redo ? history : future;
  if (!from.length || busy || restoring || drawing) return;
  const entry = from[from.length - 1];
  restoring = true; updateControls();
  try {
    const image = await loadImage(entry.data);
    to.push(snapshot(entry.layer)); from.pop();
    const ctx = entry.layer === 'image' ? paintCtx : maskCtx;
    ctx.clearRect(0, 0, base.width, base.height); ctx.drawImage(image, 0, 0);
    if (entry.layer === 'image') renderImage();
    trimHistory();
    restoring = false; setEditLayer(entry.layer);
    status(`已${redo ? '重做' : '撤销'}${entry.layer === 'image' ? '图像笔迹' : '遮罩'}修改。`);
  } catch (error) { status(error.message, 'error'); }
  finally { restoring = false; refreshMaskState(); }
}
function refreshMaskState() {
  const data = maskCtx.getImageData(0, 0, mask.width, mask.height).data;
  hasMask = false;
  for (let i = 3; i < data.length; i += 4) if (data[i]) { hasMask = true; break; }
  updateControls();
}
function point(event) {
  const rect = mask.getBoundingClientRect();
  return {x: (event.clientX - rect.left) * mask.width / rect.width,
          y: (event.clientY - rect.top) * mask.height / rect.height};
}
function showCursor(event) {
  const rect = mask.getBoundingClientRect();
  const size = Number($('brush-size').value) * rect.width / mask.width;
  Object.assign($('cursor').style, {left: `${event.clientX - rect.left}px`, top: `${event.clientY - rect.top}px`, width: `${size}px`, height: `${size}px`});
  $('cursor').hidden = busy || restoring || !loaded || mode === 'eyedropper';
  $('cursor').classList.toggle('restoring-brush', mode === 'eraser');
  $('cursor').style.background = editLayer === 'image' && mode === 'brush' ? `${$('paint-color').value}33` : 'transparent';
}
function stroke(a, b) {
  const ctx = editLayer === 'image' ? paintCtx : maskCtx;
  ctx.globalCompositeOperation = mode === 'eraser' ? 'destination-out' : 'source-over';
  ctx.strokeStyle = ctx.fillStyle = editLayer === 'image' ? $('paint-color').value : '#ee639b';
  ctx.lineWidth = Number($('brush-size').value);
  ctx.lineCap = ctx.lineJoin = 'round';
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  ctx.beginPath(); ctx.arc(b.x, b.y, ctx.lineWidth / 2, 0, Math.PI * 2); ctx.fill();
  ctx.globalCompositeOperation = 'source-over';
}
mask.addEventListener('pointerdown', event => {
  if (!loaded || busy || restoring || drawing || event.button !== 0) return;
  event.preventDefault();
  if (mode === 'eyedropper') {
    const p = point(event);
    const rgb = baseCtx.getImageData(Math.max(0, Math.min(base.width - 1, Math.floor(p.x))), Math.max(0, Math.min(base.height - 1, Math.floor(p.y))), 1, 1).data;
    setPaintColor('#' + Array.from(rgb).slice(0, 3).map(value => value.toString(16).padStart(2, '0')).join(''));
    setMode('brush'); showCursor(event);
    return;
  }
  if (editLayer === 'mask') { $('show-mask').checked = true; syncMaskVisibility(); }
  remember(); drawing = true; pointerId = event.pointerId; mask.setPointerCapture(pointerId);
  lastPoint = point(event); stroke(lastPoint, lastPoint); showCursor(event);
  if (editLayer === 'image') renderImage();
  updateControls();
});
mask.addEventListener('pointermove', event => {
  showCursor(event);
  if (!drawing || event.pointerId !== pointerId) return;
  const samples = event.getCoalescedEvents?.();
  for (const sample of (samples?.length ? samples : [event])) {
    const next = point(sample); stroke(lastPoint, next); lastPoint = next;
  }
  if (editLayer === 'image') renderImage();
});
function endStroke(event) {
  if (!drawing || (event && event.pointerId !== pointerId)) return;
  drawing = false;
  if (mask.hasPointerCapture(pointerId)) mask.releasePointerCapture(pointerId);
  pointerId = null;
  if (editLayer === 'mask') refreshMaskState(); else updateControls();
}
['pointerup', 'pointercancel', 'lostpointercapture'].forEach(type => mask.addEventListener(type, endStroke));
mask.addEventListener('pointerleave', () => $('cursor').hidden = true);
$('brush').onclick = () => setMode('brush');
$('eraser').onclick = () => setMode('eraser');
$('eyedropper').onclick = () => setMode('eyedropper');
$('edit-image').onclick = () => setEditLayer('image');
$('edit-mask').onclick = () => setEditLayer('mask');
$('paint-color').oninput = () => { setPaintColor($('paint-color').value); setMode('brush'); };
document.querySelectorAll('.swatch').forEach(button => button.onclick = () => { setPaintColor(button.dataset.color); setMode('brush'); });
$('brush-size').oninput = () => $('brush-value').textContent = `${$('brush-size').value} px`;
$('clear').onclick = () => {
  if (busy || restoring || drawing || !loaded || (editLayer === 'mask' && !hasMask)) return;
  remember();
  if (editLayer === 'image') { paintCtx.clearRect(0, 0, paint.width, paint.height); renderImage(); updateControls(); }
  else { maskCtx.clearRect(0, 0, mask.width, mask.height); refreshMaskState(); }
  status(editLayer === 'image' ? '已清除手绘笔迹，保留当前遮罩。可以撤销。' : '已清空遮罩，保留图像修改。可以撤销。');
};
$('undo').onclick = () => restoreHistory();
$('redo').onclick = () => restoreHistory(true);
$('show-mask').onchange = syncMaskVisibility;
$('file-input').onchange = event => openFile(event.target.files[0]);
['dragenter', 'dragover'].forEach(type => $('stage').addEventListener(type, event => { event.preventDefault(); if (!busy) $('stage').classList.add('dragover'); }));
$('stage').addEventListener('dragleave', () => $('stage').classList.remove('dragover'));
$('stage').addEventListener('drop', event => { event.preventDefault(); $('stage').classList.remove('dragover'); openFile(event.dataTransfer.files[0]); });
document.addEventListener('keydown', event => {
  if (event.target.matches('input,textarea,select') || busy || restoring || drawing || !loaded) return;
  const key = event.key.toLowerCase();
  if ((event.ctrlKey || event.metaKey) && ['z', 'y'].includes(key)) { event.preventDefault(); restoreHistory(key === 'y' || event.shiftKey); }
  else if (!event.ctrlKey && !event.metaKey && !event.altKey) {
    if (key === 'b') setMode('brush');
    if (key === 'e') setMode('eraser');
    if (key === 'i') setMode('eyedropper');
    if (key === 'p') setEditLayer('image');
    if (key === 'm') setEditLayer('mask');
  }
});
new ResizeObserver(layoutCanvas).observe($('stage'));

function exportMask() {
  const output = document.createElement('canvas'); output.width = mask.width; output.height = mask.height;
  const ctx = output.getContext('2d');
  ctx.drawImage(mask, 0, 0); ctx.globalCompositeOperation = 'source-in';
  ctx.fillStyle = 'white'; ctx.fillRect(0, 0, output.width, output.height);
  ctx.globalCompositeOperation = 'destination-over'; ctx.fillStyle = 'black'; ctx.fillRect(0, 0, output.width, output.height);
  return output.toDataURL('image/png');
}
function download(url, name) { const link = document.createElement('a'); link.href = url; link.download = name; link.click(); }
$('save-mask').onclick = () => download(exportMask(), 'vibepainting-mask.png');
$('save-image').onclick = () => download(base.toDataURL('image/png'), 'vibepainting-canvas.png');
$('download').onclick = () => download(resultUrl, `vibepainting-${Date.now()}.png`);
$('compare').onpointerdown = event => { if (!resultSource) return; event.currentTarget.setPointerCapture(event.pointerId); $('result-image').src = resultSource; };
['pointerup', 'pointercancel', 'lostpointercapture', 'blur'].forEach(type => $('compare').addEventListener(type, () => { if (resultUrl) $('result-image').src = resultUrl; }));
$('compare').onkeydown = event => { if ([' ', 'Enter'].includes(event.key) && resultSource) { event.preventDefault(); $('result-image').src = resultSource; } };
$('compare').onkeyup = () => { if (resultUrl) $('result-image').src = resultUrl; };
$('use-result').onclick = async () => {
  if (!resultUrl || busy || restoring) return;
  restoring = true; updateControls();
  try { installImage(await loadImage(resultUrl), '补全结果'); }
  catch (error) { status(error.message, 'error'); }
  finally { restoring = false; updateControls(); }
};
async function request(url, options) {
  let response;
  try {
    response = await fetch(url, {...options, signal: AbortSignal.timeout(15000)});
  } catch (error) {
    // A failed POST does not establish that the server is down. In particular,
    // Windows may reset an upload rejected before its body has been consumed.
    if (options?.method === 'POST') {
      let reachable = false;
      try {
        const health = await fetch('/api/health', {signal: AbortSignal.timeout(3000)});
        reachable = health.ok;
      } catch (_) { /* Report the connection failure below. */ }
      if (reachable) {
        $('connection').textContent = '● 本地服务已连接';
        throw new Error('本地服务仍在运行，但本次上传未完成。请查看 logs/server.log 中的上传错误后重试；当前画布和遮罩已保留。');
      }
    }
    $('connection').textContent = '● 本地服务连接中断';
    if (error.name === 'TimeoutError') throw new Error('请求超时。请检查运行服务的终端和 logs/server.log；确认生成状态后再重试。');
    throw new Error('无法连接本地服务，连接可能已断开或服务已停止。请检查运行服务的终端及 logs/server.log，重新启动 start_web.ps1 后再试；当前画布已保留。');
  }
  $('connection').textContent = '● 本地服务已连接';
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `请求失败 (${response.status})`);
  return body;
}
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function pollJob(id, source) {
  let failures = 0;
  for (;;) {
    let job;
    try { job = await request(`/api/jobs/${id}`); failures = 0; }
    catch (error) { if (++failures >= 5) throw error; status('连接暂时中断，正在重新获取生成状态…', 'busy'); await pause(2000); continue; }
    if (job.status === 'error') throw new Error(job.message);
    if (job.status === 'done') {
      resultUrl = job.result; resultSource = source;
      $('result-image').src = resultUrl; $('result-image').hidden = false; $('result-empty').hidden = true;
      $('result-meta').textContent = `${job.seconds} s`;
      $('result-seed').textContent = `${job.model_label || 'Lightning'}${job.seed == null ? ' · 无随机种子' : ` · Seed ${job.seed}`}`;
      status(`${job.model_label || '补全'}完成，用时 ${job.seconds} 秒。可以下载图片，或用此结果继续编辑。`);
      return;
    }
    status(job.message, 'busy'); await pause(1000);
  }
}
$('generate-form').onsubmit = async event => {
  event.preventDefault();
  if (!loaded || !hasMask || busy || restoring || drawing) return;
  const spec = modelSpec();
  if (!spec?.available) return status('当前模型不可用，请重新选择模型。', 'error');
  const steps = Number($('steps').value), strength = Number($('strength').value);
  if (spec.parameters.includes('strength') && Math.floor(steps * strength) < 1) return status('重绘强度 × 推理步数必须至少为 1，请调整生成参数。', 'error');
  busy = true; updateControls(); $('cursor').hidden = true;
  status('正在提交画面和遮罩…', 'busy');
  try {
    const source = base.toDataURL('image/png');
    const body = {image: source, mask: exportMask(), model: spec.id,
      prompt: spec.supports_prompt ? $('prompt').value : ''};
    for (const name of spec.parameters) {
      const value = $(parameterFields[name]).value;
      body[name] = name === 'negative_prompt' ? value : name === 'seed' && value === '' ? null : Number(value);
    }
    const serialized = JSON.stringify(body);
    const health = await request('/api/health');
    // Older running servers only accepted 32 MiB; negotiate before uploading.
    const limit = health.max_request_bytes || 32 * 1024 * 1024;
    const size = new Blob([serialized]).size;
    if (size > limit) throw new Error(`上传数据为 ${(size / 1048576).toFixed(1)} MB，超过当前服务的 ${(limit / 1048576).toFixed(0)} MB 上限。请缩小图片或启动新版服务；画布和遮罩已保留。`);
    const job = await request('/api/generate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: serialized});
    sessionStorage.setItem('vibepainting-job', job.id);
    await pollJob(job.id, source);
    sessionStorage.removeItem('vibepainting-job');
  } catch (error) { status(error.message, 'error'); }
  finally { busy = false; updateControls(); }
};
async function initialize() {
  try {
    await request('/api/health'); $('connection').textContent = '● 本地服务已连接';
    const catalog = await request('/api/models');
    modelCatalog = catalog.models;
    $('model').replaceChildren();
    for (const spec of modelCatalog) {
      const option = document.createElement('option'); option.value = spec.id;
      option.textContent = spec.label + (spec.available ? '' : '（模型文件未就绪）');
      option.disabled = !spec.available; $('model').append(option);
    }
    let previous;
    try { previous = localStorage.getItem('vibepainting-model'); } catch (_) { /* Use default. */ }
    const initial = modelCatalog.find(spec => spec.id === previous && spec.available)
      || modelCatalog.find(spec => spec.id === catalog.default_model && spec.available)
      || modelCatalog.find(spec => spec.available);
    if (!initial) throw new Error('没有可用的本地补全模型，请检查模型文件。');
    selectModel(initial.id);
    const id = sessionStorage.getItem('vibepainting-job');
    if (id) {
      busy = true; updateControls(); status('正在恢复上次生成任务…', 'busy');
      try { await pollJob(id, null); } finally { sessionStorage.removeItem('vibepainting-job'); busy = false; updateControls(); }
    }
  } catch (error) { $('connection').textContent = '服务连接或任务恢复失败'; status(error.message, 'error'); }
}
initialize();
