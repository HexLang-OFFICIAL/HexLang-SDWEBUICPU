import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Load the model with no safety checker
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None  # disables NSFW filtering
)

# Force CPU use (for CPU-only systems)
pipe = pipe.to("cpu")

def generate(prompt):
    with torch.autocast("cpu"):
        image = pipe(prompt, guidance_scale=7.5).images[0]
    return image

# Gradio UI
gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Prompt", placeholder="A fantasy castle on a mountain at sunset"),
    outputs=gr.Image(type="pil"),
    title="Minimal Stable Diffusion (CPU)",
    description="Lightweight Stable Diffusion WebUI for CPU, NSFW allowed"
).launch()
