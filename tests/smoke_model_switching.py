"""Run against the real local server; exercises all models and repeated switches."""
import base64
import io
import json
import subprocess
import time
from pathlib import Path
import requests
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'test_artifacts/model-switching'
URL='http://127.0.0.1:8765'


def encode(image):
    b=io.BytesIO();image.save(b,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not requests.get(URL+'/api/health',timeout=5).json()['busy']
    catalog=requests.get(URL+'/api/models',timeout=5).json()
    assert all(m['available'] for m in catalog['models'])
    original=Image.open(ROOT/'output_img2img_7513.png').convert('RGB').resize((512,512))
    mask=Image.new('L',original.size);ImageDraw.Draw(mask).rectangle((210,256,333,367),fill=255)
    original.save(OUT/'original.png');mask.save(OUT/'mask.png')
    payload=dict(image=encode(original),mask=encode(mask),prompt='an orange bird chest with soft feathers',seed=42,
                 negative_prompt='low quality, bad anatomy, blurry')
    records=[]
    for index,model in enumerate(['sd15_inpaint','sdxl_inpaint','lama','lightning','sd15_inpaint','sd15_inpaint']):
        response=requests.post(URL+'/api/generate',json={**payload,'model':model},timeout=15)
        assert response.status_code==202,response.text
        job_id=response.json()['id']; deadline=time.monotonic()+180
        while True:
            job=requests.get(URL+'/api/jobs/'+job_id,timeout=10).json()
            if job['status']!='running':break
            assert time.monotonic()<deadline,'Job timed out'
            time.sleep(.5)
        assert job['status']=='done',job
        assert job['model']==model
        assert (job['seed'] is None) == (model=='lama')
        image=Image.open(io.BytesIO(base64.b64decode(job['result'].split(',')[1]))).convert('RGB')
        assert image.size==original.size
        outside=np.asarray(mask)==0
        assert np.array_equal(np.asarray(image)[outside],np.asarray(original)[outside])
        assert not np.array_equal(np.asarray(image),np.asarray(original))
        image.save(OUT/f'{index}_{model}.png')
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
        row=dict(model=model,seconds=job['seconds'],seed=job['seed'],gpu_memory_used_mib=int(memory),outside_unchanged=True)
        records.append(row);print(json.dumps(row),flush=True)
        (OUT/'real_generation.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    assert (OUT/'4_sd15_inpaint.png').read_bytes()==(OUT/'5_sd15_inpaint.png').read_bytes(), 'Same cached model + seed must be repeatable'
    print('All model routes, repeated switches, unchanged outside pixels and cached repeat passed.',flush=True)


if __name__=='__main__':main()
