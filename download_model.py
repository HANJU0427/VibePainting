"""Download the full SDXL-Lightning 2-step checkpoint into this project.

Uses the user's HTTP_PROXY / HTTPS_PROXY settings, if present.
"""
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    target = Path(__file__).resolve().parent / 'models'
    target.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id='ByteDance/SDXL-Lightning',
        filename='sdxl_lightning_2step.safetensors',
        local_dir=str(target),
    )
    print(f'Lightning checkpoint ready: {path}')


if __name__ == '__main__':
    main()
