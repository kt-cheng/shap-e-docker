import torch
from IPython.display import display

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Load model transmitter")
xm = load_model('transmitter', device=device)

print("Load model text300M")
model_text = load_model('text300M', device=device)

diffusion_text = diffusion_from_config(load_config('diffusion'))

batch_size = 3
guidance_scale = 15.0
prompt = "A chair that looks like an avocado"

latents_text = sample_latents(
    batch_size=batch_size,
    model=model_text,
    diffusion=diffusion_text,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)

render_mode = 'nerf' # you can change this to 'stf'
size = 64 # this is the size of the renders; higher values take longer to render.

cameras_text = create_pan_cameras(size, device)
for i, latent in enumerate(latents_text):
    images = decode_latent_images(xm, latent, cameras_text, rendering_mode=render_mode)
    display(gif_widget(images))

for i, latent in enumerate(latents_text):
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'example_mesh_text_{i}.obj', 'w') as f:
        t.write_obj(f)
