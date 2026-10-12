"""
Download ControlNet Inpainting model for SDXL
Model: destitech/controlnet-inpaint-dreamer-sdxl
"""

from pathlib import Path


def main():
    # Share the inference entry point's optional-import workaround on Windows.
    from sdxl_lightning_inpaint import ControlNetModel, torch

    save_dir = Path(__file__).resolve().parent / 'models/controlnet-inpaint-sdxl'
    print('Downloading ControlNet Inpainting model for SDXL...', flush=True)
    controlnet = ControlNetModel.from_pretrained(
        "destitech/controlnet-inpaint-dreamer-sdxl",
        torch_dtype=torch.float16,
    )
    controlnet.save_pretrained(save_dir)
    print(f'ControlNet model ready: {save_dir}')


if __name__ == '__main__':
    main()
