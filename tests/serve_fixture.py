"""Browser-test fixture only. This does not run a real AI model."""
import sys
import time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from web_server import AppServer
from model_catalog import MODELS


def fixture(image, mask, progress, **params):
    progress('测试替代推理正在运行')
    time.sleep(0.5)
    return Image.composite(Image.new('RGB', image.size, '#176a73'), image, mask)


if __name__ == '__main__':
    print('TEST FIXTURE ONLY: http://127.0.0.1:8767', flush=True)
    # UI regression tests must work in a clone without downloaded model weights.
    catalog = [{'id': key, **spec, 'available': True} for key, spec in MODELS.items()]
    with patch('web_server.require_available'), patch('web_server.list_models', return_value=catalog):
        AppServer(('127.0.0.1', 8767), fixture).serve_forever()
