import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin

url = "http://127.0.0.1:7860"

def txt2img(model,
    enable_hr=False, denoising_strength=0, firstphase_width=0, firstphase_height=0, hr_scale=2, hr_upscaler="Latent", hr_second_pass_steps=0, hr_resize_x=0, hr_resize_y=0,
    prompt="", seed=1, sampler_name="Euler a", batch_size=1, n_iter=1, steps=42, cfg_scale=13, width=768, height=768, restore_faces=False, tiling=False, negative_prompt=""):
    payload = {
    "enable_hr": enable_hr,
    "denoising_strength": denoising_strength,
    "firstphase_width": firstphase_width,
    "firstphase_height": firstphase_height,
    "hr_scale": hr_scale,
    "hr_upscaler": hr_upscaler,
    "hr_second_pass_steps": hr_second_pass_steps,
    "hr_resize_x": hr_resize_x,
    "hr_resize_y": hr_resize_y,
    "prompt": prompt,
    "seed": seed,
    "sampler_name": sampler_name,
    "batch_size": batch_size,
    "n_iter": n_iter,
    "steps": steps,
    "cfg_scale": cfg_scale,
    "width": width,
    "height": height,
    "restore_faces": restore_faces,
    "tiling": tiling,
    "negative_prompt": negative_prompt,
    "override_settings": {"sd_model_checkpoint":model},
    "override_settings_restore_afterwards": False
    }

    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=json.loads(json.dumps(payload)))
    print(response)
    r = response.json()

    images = []
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        images.append(image)

    return images

def refresh_models():
    requests.post(url=f'{url}/sdapi/v1/refresh-checkpoints')

def set_model(model):
    payload = {"sd_model_checkpoint": model}
    requests.post(url=f'{url}/sdapi/v1/options', json=payload)
