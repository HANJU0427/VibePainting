import base64
import http.client
import io
import json
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
from web_server import AppServer, validate_request
from model_catalog import MODELS


def encode(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def payload():
    image = Image.new('RGB', (160, 80), (30, 60, 90))
    mask = Image.new('L', image.size)
    ImageDraw.Draw(mask).rectangle((60, 20, 100, 60), fill=255)
    return {'image': encode(image), 'mask': encode(mask), 'prompt': 'flowers', 'seed': 42}


class ValidationTests(unittest.TestCase):
    def setUp(self):
        availability = patch('web_server.require_available')
        availability.start()
        self.addCleanup(availability.stop)

    def test_model_defaults_and_unused_parameters(self):
        for model, steps, guidance in [('lightning', 2, 0), ('sd15_inpaint', 30, 7.5), ('sdxl_inpaint', 20, 8)]:
            with self.subTest(model=model):
                _, _, params = validate_request({**payload(), 'model': model})
                self.assertEqual(params['model'], model)
                self.assertEqual(params['num_inference_steps'], steps)
                self.assertEqual(params['guidance_scale'], guidance)
                self.assertEqual('controlnet_conditioning_scale' in params, model == 'lightning')
        _, _, params = validate_request({**payload(), 'model': 'lama', 'strength': 0, 'num_inference_steps': 'ignored'})
        self.assertIsNone(params['seed'])
        self.assertEqual(params['prompt'], '')
        self.assertNotIn('strength', params)

    def test_unknown_or_missing_model_rejected(self):
        for model in ['unknown', '../models', None, []]:
            with self.subTest(model=model), self.assertRaises(ValueError):
                validate_request({**payload(), 'model': model})
        with patch('web_server.require_available', side_effect=ValueError('模型文件不完整')):
            with self.assertRaisesRegex(ValueError, '模型文件不完整'):
                validate_request(payload())

    def test_empty_mask_and_mismatched_size(self):
        for mask in [Image.new('L', (160, 80)), Image.new('L', (80, 80), 255)]:
            with self.assertRaises(ValueError):
                validate_request({**payload(), 'mask': encode(mask)})

    def test_invalid_parameters(self):
        for key, value in [('prompt', None), ('prompt', 'a' * 4001), ('strength', float('nan')), ('strength', 0.1),
                           ('num_inference_steps', 2.5), ('seed', -1), ('guidance_scale', 'abc')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_request({**payload(), key: value})

    def test_random_seed_and_valid_dimensions(self):
        data = payload()
        data['seed'] = None
        image, mask, params = validate_request(data)
        self.assertEqual(image.size, mask.size)
        self.assertIsInstance(params['seed'], int)
        for prompt in ['', '   \n\t']:
            _, _, params = validate_request({**data, 'prompt': prompt})
            self.assertEqual(params['prompt'], '')
        data.pop('prompt')
        self.assertEqual(validate_request(data)[2]['prompt'], '')


class ServerTests(unittest.TestCase):
    def setUp(self):
        availability = patch('web_server.require_available')
        availability.start()
        self.addCleanup(availability.stop)
        self.release = threading.Event()
        def generate(image, mask, progress, **params):
            self.received_params = params
            progress('test inference')
            self.release.wait(5)
            return Image.composite(Image.new('RGB', image.size, 'red'), image, mask)
        self.server = AppServer(('127.0.0.1', 0), generate)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, data=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        connection.request(method, path, json.dumps(data) if data is not None else None,
                           headers or {'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read()
        result = json.loads(raw) if 'application/json' in response.getheader('Content-Type') else raw
        connection.close()
        return response.status, result

    def wait_job(self, job_id):
        for _ in range(100):
            code, body = self.request('GET', '/api/jobs/' + job_id)
            if body['status'] != 'running':
                return body
            time.sleep(0.02)
        self.fail('job timed out')

    def test_generation_busy_and_download(self):
        code, body = self.request('POST', '/api/generate', {**payload(), 'prompt': ''})
        self.assertEqual(code, 202)
        self.assertEqual(self.request('POST', '/api/generate', payload())[0], 409)
        self.release.set()
        result = self.wait_job(body['id'])
        self.assertEqual(result['status'], 'done')
        self.assertEqual(result['seed'], 42)
        self.assertEqual(result['model'], 'lightning')
        self.assertEqual(self.received_params['prompt'], '')
        image = Image.open(io.BytesIO(base64.b64decode(result['result'].split(',')[1])))
        self.assertEqual(image.size, (160, 80))
        self.assertEqual(image.getpixel((0, 0)), (30, 60, 90))
        self.assertEqual(image.getpixel((80, 40)), (255, 0, 0))

    def test_catalog_and_selected_model_reach_worker(self):
        code, catalog = self.request('GET', '/api/models')
        self.assertEqual(code, 200)
        self.assertEqual({m['id'] for m in catalog['models']}, set(MODELS))
        self.release.set()
        for model in ['sd15_inpaint', 'lama', 'sdxl_inpaint']:
            code, job = self.request('POST', '/api/generate', {**payload(), 'model': model})
            self.assertEqual(code, 202)
            result = self.wait_job(job['id'])
            self.assertEqual(result['status'], 'done')
            self.assertEqual(result['model'], model)
            self.assertEqual(result['model_label'], MODELS[model]['label'])
            self.assertEqual(self.received_params['model'], model)


    def test_error_propagation(self):
        def fail(*args, **kwargs):
            raise RuntimeError('test model unavailable')
        self.server.generator = fail
        code, body = self.request('POST', '/api/generate', payload())
        self.assertEqual(code, 202)
        result = self.wait_job(body['id'])
        self.assertEqual(result['status'], 'error')
        self.assertIn('test model unavailable', result['message'])

    def test_second_server_cannot_bind_same_port(self):
        with self.assertRaises(OSError):
            second = AppServer(('127.0.0.1', self.server.server_port))
            second.server_close()

    def test_oversized_upload_returns_http_error(self):
        with patch('web_server.MAX_BODY', 1024):
            code, body = self.request('POST', '/api/generate', {'padding': 'x' * 4096})
        self.assertEqual(code, 413)
        self.assertIn('上传上限', body['error'])
        self.assertEqual(self.request('GET', '/api/health')[0], 200)

    def test_static_validation_and_origin(self):
        self.assertEqual(self.request('GET', '/')[0], 200)
        self.assertEqual(self.request('GET', '/favicon.ico')[0], 204)
        self.assertEqual(self.request('GET', '/sdxl_lightning_inpaint.py')[0], 404)
        self.assertEqual(self.request('POST', '/api/generate', {})[0], 400)
        self.assertEqual(self.request('POST', '/api/generate', payload(),
                         {'Content-Type': 'application/json', 'Origin': 'https://example.com'})[0], 403)


class ImagePipelineTests(unittest.TestCase):
    def test_aspect_ratio_and_untouched_pixels(self):
        import sdxl_lightning_inpaint as engine
        calls = []
        class FakePipeline:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return type('Result', (), {'images': [Image.new('RGB', (1024, 1024), 'red')]})()
        image = Image.new('RGB', (160, 80), (30, 60, 90))
        mask = Image.new('L', image.size)
        ImageDraw.Draw(mask).rectangle((60, 20, 100, 60), fill=255)
        with patch.object(engine, '_pipe', FakePipeline()):
            result = engine.inpaint_image(image, mask, 'flowers')
        self.assertEqual(result.size, image.size)
        self.assertEqual(calls[0]['image'].getpixel((0, 0)), (127, 127, 127))
        self.assertEqual(calls[0]['image'].getpixel((0, 256)), (30, 60, 90))
        self.assertEqual(calls[0]['mask_image'].getpixel((512, 512)), 255)
        self.assertEqual(calls[0]['control_image'].getpixel((512, 512)), (255, 255, 255))
        self.assertEqual(calls[0]['control_image'].getpixel((0, 256)), (30, 60, 90))
        self.assertEqual(result.getpixel((80, 40)), (255, 0, 0))
        for y in range(image.height):
            for x in range(image.width):
                if mask.getpixel((x, y)) == 0:
                    self.assertEqual(result.getpixel((x, y)), image.getpixel((x, y)))


if __name__ == '__main__':
    unittest.main()
