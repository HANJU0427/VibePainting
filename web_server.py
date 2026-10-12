"""Local mask editor: python web_server.py [--port 8765]."""
import argparse
import base64
import binascii
import io
import json
import faulthandler
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError
from model_catalog import DEFAULT_MODEL, get_model, list_models, require_available

ROOT = Path(__file__).resolve().parent
# Two RGBA PNGs at the 16-megapixel limit can exceed 32 MiB after base64
# encoding. Keep lossless pixels and allow both images plus JSON overhead.
MAX_BODY = 192 * 1024 * 1024
MAX_PIXELS = 16_000_000
logger = logging.getLogger('vibepainting')


def configure_logging():
    """Keep diagnostics even after a terminal is closed or a native library fails."""
    log_dir = ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)
    handler = RotatingFileHandler(log_dir / 'server.log', maxBytes=2_000_000,
                                  backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # Keep this handle open for the lifetime of the process.
    native_log = (log_dir / 'native-crash.log').open('a', encoding='utf-8')
    faulthandler.enable(file=native_log)
    return native_log


def decode_image(value):
    if not isinstance(value, str) or not value.startswith('data:image/') or ';base64,' not in value:
        raise ValueError('图片格式无效，请重新选择图片')
    try:
        raw = base64.b64decode(value.split(',', 1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError('图片过大，请使用不超过 1600 万像素的图片')
            image.load()
            return image.copy()
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('无法读取图片，请使用 PNG、JPEG 或 WebP') from exc


def validate_request(data):
    if not isinstance(data, dict):
        raise ValueError('请求必须为 JSON 对象')
    model = data.get('model', DEFAULT_MODEL)
    spec = get_model(model)
    require_available(model)
    prompt = data.get('prompt', '') if spec['supports_prompt'] else ''
    negative = data.get('negative_prompt', '') if spec['supports_prompt'] else ''
    if not isinstance(prompt, str) or len(prompt) > 4000:
        raise ValueError('补全描述必须为文本，最多 4000 字（可以留空）')
    if not isinstance(negative, str) or len(negative) > 4000:
        raise ValueError('负面提示词过长')
    params = {'model': model, 'prompt': prompt.strip(), 'negative_prompt': negative}
    for name, default, low, high in [
        ('strength', 1.0, 0.01, 1), ('guidance_scale', 0, 0, 20),
        ('controlnet_conditioning_scale', 0.3, 0, 2),
        ('num_inference_steps', 2, 1, 30),
    ]:
        if name not in spec['parameters']:
            continue
        value = data.get(name, spec['defaults'][name])
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} 必须在 {low} 到 {high} 之间')
        params[name] = value
    if 'num_inference_steps' in params:
        steps = params['num_inference_steps']
        if int(steps) != steps:
            raise ValueError('推理步数必须为整数')
        params['num_inference_steps'] = int(steps)
        if int(steps * params['strength']) < 1:
            raise ValueError('重绘强度 × 推理步数必须至少为 1')
    seed = data.get('seed') if 'seed' in spec['parameters'] else None
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 4294967295):
        raise ValueError('种子必须为 0 到 4294967295 的整数，或留空')
    params['seed'] = (secrets.randbelow(4294967296) if seed is None else seed) if 'seed' in spec['parameters'] else None
    image = decode_image(data.get('image')).convert('RGB')
    mask = decode_image(data.get('mask')).convert('L')
    if image.size != mask.size:
        raise ValueError('原图和遮罩尺寸必须一致')
    if mask.getbbox() is None:
        raise ValueError('请先绘制需要补全的区域')
    return image, mask, params


class AppServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def server_bind(self):
        # HTTPServer enables SO_REUSEADDR by default. On Windows that can let
        # two live servers bind the same port and route requests to old code.
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request, client_address):
        logger.exception('HTTP request failed from %s', client_address)

    def __init__(self, address, generator=None):
        super().__init__(address, Handler)
        self.job = None
        self.job_lock = threading.Lock()
        self.generator = generator

    def generate(self, job_id, image, mask, params):
        started = time.monotonic()
        logger.info('Job %s started: model=%s, image=%s, empty_prompt=%s', job_id, params['model'], image.size, not params['prompt'])
        def progress(message):
            logger.info('Job %s: %s', job_id, message)
            with self.job_lock:
                self.job['message'] = message
        try:
            generate = self.generator
            if generate is None:
                from inpaint_engine import inpaint_image
                generate = inpaint_image
            result = generate(image, mask, **params, progress=progress)
            buffer = io.BytesIO()
            result.save(buffer, format='PNG')
            with self.job_lock:
                self.job.update(status='done', message='补全完成',
                                seconds=round(time.monotonic() - started, 1),
                                result='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode())
            logger.info('Job %s completed in %.1fs', job_id, time.monotonic() - started)
        except Exception as exc:
            logger.exception('Job %s failed', job_id)
            with self.job_lock:
                self.job.update(status='error', message=f'生成失败：{str(exc)[:1200]}')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Do not depend on an attached terminal for serving HTTP requests.
        logger.info('%s %s', self.client_address[0], format % args)

    def send(self, status, content, content_type='application/json; charset=utf-8'):
        if not isinstance(content, bytes):
            content = json.dumps(content, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        port = self.server.server_port
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host') not in allowed:
            return False
        origin = self.headers.get('Origin')
        return origin is None or origin in {f'http://{host}' for host in allowed}

    def do_GET(self):
        if not self.local_request():
            return self.send(403, {'error': '仅支持本机访问'})
        path = urlparse(self.path).path
        assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                  '/styles.css': ('styles.css', 'text/css; charset=utf-8')}
        if path in assets:
            name, mime = assets[path]
            return self.send(200, (ROOT / 'web' / name).read_bytes(), mime)
        if path == '/api/health':
            with self.server.job_lock:
                busy = bool(self.server.job and self.server.job['status'] == 'running')
            return self.send(200, {'status': 'ok', 'pid': os.getpid(), 'busy': busy,
                                   'max_request_bytes': MAX_BODY})
        if path == '/favicon.ico':
            return self.send(204, b'', 'image/x-icon')
        if path == '/api/models':
            return self.send(200, {'default_model': DEFAULT_MODEL, 'models': list_models()})
        if path.startswith('/api/jobs/'):
            job_id = path.rsplit('/', 1)[-1]
            with self.server.job_lock:
                job = self.server.job
                if job and job['id'] == job_id:
                    job = dict(job)
                else:
                    job = None
            return self.send(200, job) if job else self.send(404, {'error': '任务不存在或已被后续任务替换'})
        return self.send(404, {'error': '未找到页面'})

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.send(400, {'error': '无效的请求长度'})
        if length <= 0:
            return self.send(400, {'error': '请求内容不能为空'})
        if length > MAX_BODY:
            logger.warning('Upload rejected: %s bytes exceeds %s bytes', length, MAX_BODY)
            self.send(413, {'error': f'图片数据过大，上传上限为 {MAX_BODY // (1024 * 1024)} MB'})
            # Send the error before draining. Closing a socket with unread
            # upload bytes on Windows can replace HTTP 413 with a TCP reset.
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_WR)
            self.connection.settimeout(2)
            remaining = min(length, MAX_BODY)
            deadline = time.monotonic() + 2
            try:
                while remaining and time.monotonic() < deadline:
                    chunk = self.rfile.read1(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            except OSError:
                pass
            self.close_connection = True
            return
        # Consume bounded bodies before an early rejection, otherwise Windows
        # may reset the socket before the client receives the error response.
        self.connection.settimeout(30)
        try:
            raw = self.rfile.read(length)
        except TimeoutError:
            return self.send(408, {'error': '上传超时，请重试'})
        if not self.local_request():
            return self.send(403, {'error': '仅支持本机访问'})
        if self.path != '/api/generate':
            return self.send(404, {'error': '接口不存在'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.send(415, {'error': '请使用 JSON 请求'})
        try:
            data = json.loads(raw)
            image, mask, params = validate_request(data)
        except (ValueError, TypeError, UnicodeError) as exc:
            return self.send(400, {'error': str(exc)})
        with self.server.job_lock:
            if self.server.job and self.server.job['status'] == 'running':
                return self.send(409, {'error': '已有生成任务正在运行，请稍后重试'})
            job_id = secrets.token_hex(12)
            self.server.job = {'id': job_id, 'status': 'running', 'message': '正在准备模型…',
                               'seed': params['seed'], 'model': params['model'],
                               'model_label': get_model(params['model'])['label']}
        threading.Thread(target=self.server.generate, args=(job_id, image, mask, params), daemon=True).start()
        return self.send(202, {'id': job_id})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    native_log = configure_logging()
    logger.info('Starting server PID %s on port %s', os.getpid(), args.port)
    try:
        server = AppServer(('127.0.0.1', args.port))
    except OSError as exc:
        logger.exception('Cannot bind port %s', args.port)
        raise SystemExit(f'Cannot start on port {args.port}: {exc}. Stop the existing server or use --port with another port.')
    # Initialize native ML libraries on the main thread on Windows. Weights
    # remain lazy-loaded by the generation worker on the first request.
    print('Initializing PyTorch and Diffusers...', flush=True)
    try:
        from inpaint_engine import inpaint_image
        server.generator = inpaint_image
    except Exception:
        server.server_close()
        logger.exception('Server startup failed')
        raise
    print(f'VibePainting: http://127.0.0.1:{args.port}', flush=True)
    print('Press Ctrl+C to stop. Model loads on the first generation.', flush=True)
    print(f'Logs: {ROOT / "logs/server.log"}', flush=True)
    logger.info('Server ready: http://127.0.0.1:%s', args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info('Server stopped')
        server.server_close()
