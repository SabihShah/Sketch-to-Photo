import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import PidiNetDetector
import matplotlib.pyplot as plt


def load_pipeline(device="cuda"):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_softedge", torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()  # helps fit in 8GB VRAM
    pipe.enable_xformers_memory_efficient_attention()

    return pipe


def get_softedge_map(image_path, size=512):
    processor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    img = Image.open(image_path).convert("RGB").resize((size, size))
    edge_map = processor(img)
    return edge_map


def run_inference(
    pipe,
    edge_map,
    prompt,
    negative_prompt="",
    steps=35,
    guidance_scale=7.5,
    conditioning_scale=1.2,
    seed=42,
):
    generator = torch.manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=edge_map,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=conditioning_scale,
        generator=generator,
    ).images[0]
    return output


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", required=True, help="Path to preprocessed sketch")
    # parser.add_argument("--output", required=True, help="Path to save generated photo")
    # parser.add_argument("--prompt", required=True, help="Text prompt describing desired output")
    # parser.add_argument("--negative_prompt", default="lowres, blurry, deformed, bad anatomy")
    # parser.add_argument("--steps", type=int, default=25)
    # parser.add_argument("--size", type=int, default=512)
    # args = parser.parse_args()

    input = "output-thresh.jpeg"
    output = "final-output.jpeg"
    size = 512
    prompt = "photorealistic old stone house, detailed brick and wood texture, natural daylight, high resolution photograph"
    negative_prompt = "lowres, blurry, deformed, bad anatomy"
    steps = 25

    pipe = load_pipeline()
    edge_map = get_softedge_map(input, size=size)
    result = run_inference(pipe, edge_map, prompt, negative_prompt, steps=steps)

    plt.imshow(result)
    plt.axis("off")

    result.save(output)
    print(f"Saved generated image: {output}")
