import os
import torch
import argparse
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


class SketchPhotoDataset(Dataset):
    def __init__(self, data_dir, size=512):
        self.data_dir = Path(data_dir)
        self.image_paths = sorted((self.data_dir / "images").glob("*.jpg"))
        self.cond_paths = sorted((self.data_dir / "conditioning").glob("*.jpg"))
        with open(self.data_dir / "captions.txt") as f:
            self.captions = [line.strip() for line in f.readlines()]

        assert len(self.image_paths) == len(self.cond_paths) == len(self.captions), \
            "Mismatch between images, conditioning maps, and captions"

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_paths[idx]).convert("RGB"))
        cond = self.transform(Image.open(self.cond_paths[idx]).convert("RGB"))
        caption = self.captions[idx]
        return {"pixel_values": image, "conditioning": cond, "caption": caption}


def setup_lora_unet(pipe, rank=8, alpha=16):
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.print_trainable_parameters()
    return pipe


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_softedge", torch_dtype=torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        safety_checker=None
    )
    pipe = setup_lora_unet(pipe, rank=args.rank, alpha=args.alpha)
    pipe.to(device)

    # Freeze everything except LoRA params
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.controlnet.requires_grad_(False)

    dataset = SketchPhotoDataset(args.data_dir, size=args.size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    noise_scheduler = pipe.scheduler

    pipe.unet.train()
    for epoch in range(args.epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device)
            conditioning = batch["conditioning"].to(device)
            captions = batch["caption"]

            latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            text_inputs = pipe.tokenizer(captions, padding="max_length", truncation=True,
                                          max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(device)
            encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0]

            down_res, mid_res = pipe.controlnet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=conditioning, return_dict=False
            )

            model_pred = pipe.unet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res
            ).sample

            loss = torch.nn.functional.mse_loss(model_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress.set_postfix(loss=loss.item())

    os.makedirs(args.output_dir, exist_ok=True)
    pipe.unet.save_pretrained(args.output_dir)
    print(f"Saved LoRA weights to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./lora_output")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    args = parser.parse_args()

    train(args)