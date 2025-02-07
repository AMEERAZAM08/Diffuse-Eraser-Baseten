import gradio as gr
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur
import os 


def preprocess_image(image_path, device):
    image = to_tensor((load_image(image_path)))
    image = image.unsqueeze_(0).float() * 2 - 1 # [0,1] --> [-1,1]
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (1024, 1024))
    image = image.to(dtype).to(device)
    return image

def preprocess_mask(mask_path, device):
    mask = to_tensor((load_image(mask_path, convert_method=lambda img: img.convert('L'))))
    mask = mask.unsqueeze_(0).float()  # 0 or 1
    mask = F.interpolate(mask, (1024, 1024))
    mask = gaussian_blur(mask, kernel_size=(77, 77))#default is 
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(dtype).to(device)
    return mask

def remove_objects(pipeline,edit_images):
    """
    Runs the attentive eraser pipeline. 
    Returns the processed (inpainted) PIL image.
    """
    # Fixed parameters
  

    source_image = edit_images[0]
    mask_image = edit_images[1]
    mask_image = mask_image.convert("L")
    prompt = ""
    seed=123 
    generator = torch.Generator(device=device).manual_seed(seed)
    source_image_ = preprocess_image(source_image, device)
    mask_ = preprocess_mask(mask_image, device)
    image = pipeline(
        prompt=prompt,
        image=source_image_,
        mask_image=mask_,
        height=1024,
        width=1024,
        AAS=True, # enable AAS
        strength=0.8, # inpainting strength
        rm_guidance_scale=9, # removal guidance scale
        ss_steps = 9, # similarity suppression steps
        ss_scale = 0.3, # similarity suppression scale
        AAS_start_step=0, # AAS start step
        AAS_start_layer=34, # AAS start layer
        AAS_end_layer=70, # AAS end layer
        num_inference_steps=50, # number of inference steps # AAS_end_step = int(strength*num_inference_steps)
        generator=generator,
        guidance_scale=1,
    ).images[0]
    final_image = image.resize((source_image.size[0], source_image.size[1]))

    return final_image