import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import PidiNetDetector


def load_pipeline(device="cuda", lora_path=None):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_softedge",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()  # helps fit in 8GB VRAM
    pipe.enable_xformers_memory_efficient_attention()

    if lora_path:
        pipe.load_lora_weights(lora_path)
        print(f"Loaded LoRA weights from {lora_path}")

    return pipe


def get_softedge_map(image_path, size=512):
    processor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    img = Image.open(image_path).convert("RGB").resize((size, size))
    edge_map = processor(img)
    return edge_map


def run_inference(pipe, edge_map, prompt, negative_prompt="", steps=35, guidance_scale=7.5,
                   conditioning_scale=1.2, seed=42):
    generator = torch.manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=edge_map,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=conditioning_scale,
        generator=generator
    ).images[0]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to preprocessed sketch")
    parser.add_argument("--output", required=True, help="Path to save generated photo")
    parser.add_argument("--prompt", required=True, help="Text prompt describing desired output")
    parser.add_argument("--negative_prompt", default="lowres, blurry, deformed, extra objects, cartoon, illustration, painting, sketch, watermark")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--conditioning_scale", type=float, default=1.2)
    parser.add_argument("--lora_path", default=None, help="Path to trained LoRA weights, if using")
    args = parser.parse_args()

    pipe = load_pipeline(lora_path=args.lora_path)
    edge_map = get_softedge_map(args.input, size=args.size)
    result = run_inference(pipe, edge_map, args.prompt, args.negative_prompt,
                            steps=args.steps, conditioning_scale=args.conditioning_scale)

    result.save(args.output)
    print(f"Saved generated image: {args.output}")