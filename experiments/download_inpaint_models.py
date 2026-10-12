"""Download pinned public comparison checkpoints into this project only."""
import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1] / 'models' / 'comparison'
REPOS = {
    'sdxl_inpaint': ('diffusers/stable-diffusion-xl-1.0-inpainting-0.1', '115134f363124c53c7d878647567d04daf26e41e'),
    'sd15_inpaint': ('stable-diffusion-v1-5/stable-diffusion-inpainting', '8a4288a76071f7280aedbdb3253bdb9e9d5d84bb'),
}
LAMA_URL = 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt'
LAMA_MD5 = 'e3aa4aaa15225a33ec84f9f4bc47e500'


def download_segment(item):
    url, start, end, target = item
    if target.exists() and target.stat().st_size == end-start+1:
        return target
    for attempt in range(5):
        try:
            with requests.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(30, 60)) as response:
                response.raise_for_status()
                if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                    raise RuntimeError('Server did not honor exact byte range')
                with target.open('wb') as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            if target.stat().st_size != end-start+1:
                raise RuntimeError('Incomplete segment')
            return target
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 + attempt)


def segmented_download(url, target, partial, size):
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise RuntimeError('Partial file exceeds remote size')
    chunks = target.parent / (target.name + '.chunks')
    chunks.mkdir(exist_ok=True)
    step = 64 * 1024 * 1024
    jobs = [(url, start, min(start+step, size)-1, chunks / f'{start}-{min(start+step, size)-1}')
            for start in range(offset, size, step)]
    print(f'Resuming {target.relative_to(ROOT)} at {offset/1e6:.0f}/{size/1e6:.0f} MB using parallel ranges', flush=True)
    # Append only ordered, fully downloaded segments. A retry can resume the prefix.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        with partial.open('ab') as output:
            for path in pool.map(download_segment, jobs):
                with path.open('rb') as data:
                    while chunk := data.read(4 * 1024 * 1024):
                        output.write(chunk)
                output.flush()
                path.unlink()
    if partial.stat().st_size != size:
        raise RuntimeError('Final model length differs from remote size')
    partial.replace(target)
    print(f'Downloaded {target.relative_to(ROOT)} ({size/1e6:.1f} MB)', flush=True)
    return {'file': str(target.relative_to(ROOT)), 'bytes': size, 'cached': False}


def download(item):
    url, target = item
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return {'file': str(target.relative_to(ROOT)), 'bytes': target.stat().st_size, 'cached': True}
    partial = target.with_name(target.name + '.partial')
    head = requests.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    size = int(head.headers.get('Content-Length', '0'))
    if size > 64 * 1024 * 1024 and head.headers.get('Accept-Ranges') == 'bytes':
        return segmented_download(url, target, partial, size)
    for attempt in range(6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            # Explicit pinned URLs avoid altering the user's global HF cache.
            with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                resume = response.status_code == 206
                if resume and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Unexpected download range')
                expected = int(response.headers.get('Content-Length', '0'))
                received = 0
                with partial.open('ab' if resume else 'wb') as handle:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        handle.write(chunk)
                        received += len(chunk)
                if expected and received != expected:
                    raise RuntimeError(f'Incomplete response: {received}/{expected}')
            partial.replace(target)
            print(f'Downloaded {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.1f} MB)', flush=True)
            return {'file': str(target.relative_to(ROOT)), 'bytes': target.stat().st_size, 'cached': False}
        except Exception as error:
            print(f'Retry {attempt+1}: {target.name}: {type(error).__name__}: {str(error)[:200]}', flush=True)
            if attempt == 5:
                raise
            time.sleep(2 + attempt * 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=[*REPOS, 'lama'], default=[*REPOS, 'lama'],
                        help='Download only selected models (default: all three).')
    selected = parser.parse_args(argv).models
    ROOT.mkdir(parents=True, exist_ok=True)
    files = []
    for name, (repo, revision) in REPOS.items():
        if name not in selected:
            continue
        response = requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}', timeout=30)
        response.raise_for_status()
        for entry in response.json()['siblings']:
            path = entry['rfilename']
            if path.endswith(('.json', '.txt', '.fp16.safetensors')) or path == 'README.md':
                files.append((f'https://huggingface.co/{repo}/resolve/{revision}/{path}', ROOT / name / path))
    if 'lama' in selected:
        files.append((LAMA_URL, ROOT / 'lama' / 'big-lama.pt'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, files))
    sources = {'repositories': {name: value for name, value in REPOS.items() if name in selected}, 'files': results}
    if 'lama' in selected:
        digest = hashlib.md5((ROOT / 'lama' / 'big-lama.pt').read_bytes()).hexdigest()
        if digest != LAMA_MD5:
            raise RuntimeError(f'LaMa checksum mismatch: {digest}')
        sources['lama'] = {'url': LAMA_URL, 'md5': digest}
    (ROOT / 'sources.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    print(f'Selected models ready: {", ".join(selected)}', flush=True)


if __name__ == '__main__':
    main()
