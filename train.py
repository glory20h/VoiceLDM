import os
import math
import yaml
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from transformers import (
    ClapModel,
    ClapProcessor,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
)

from diffusers import (
    AutoencoderKL, 
    DDIMScheduler, 
    UNet2DConditionModel
)
from diffusers.optimization import get_scheduler

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from voiceldm import (
    AudioDataset,
    CollateFn,
    TextEncoder,
    UNetWrapper, 
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=None, type=str, help="Path to configuration file (overrides all other arguments when provided)")
    parser.add_argument("--output_dir", default="outputs", help="Directory name to save output files")
    parser.add_argument("--load_from_ckpt_path", default=None, help="Path to load checkpoint from")
    parser.add_argument("--checkpoints_total_limit", type=int, default=2, help="Total limit of checkpoints")
    parser.add_argument("--mixed_precision", default=None, help="'fp16' to enable mixed precision training")
    parser.add_argument("--max_train_steps", type=int, default=1000000)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=20000)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--uncond_desc_prob", type=float, default=0.1, help="prob. to drop description condition")
    parser.add_argument("--uncond_text_prob", type=float, default=0.1, help="prob. to drop content condition")
    parser.add_argument("--add_noise_prob", type=float, default=0.5, help="prob. to add noise")
    parser.add_argument("--block_out_channels", type=list, default=[192, 384, 576, 960])

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)

    return args


def main():
    args = parse_args()

    args.output_dir = os.path.join("results", args.output_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        automatic_checkpoint_naming=True,
        total_limit=args.checkpoints_total_limit,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # Save args as yaml file
        with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(args), f)

    noise_scheduler = DDIMScheduler.from_pretrained("cvssp/audioldm-m-full", subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("cvssp/audioldm-m-full", subfolder="vae")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    text_encoder = TextEncoder(SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts"))
    text_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

    vae.requires_grad_(False)
    clap_model.requires_grad_(False)

    uncond_embed = torch.load('uncond_embed.pt')

    unet = UNet2DConditionModel(
        sample_size = 128,
        in_channels = 8,
        out_channels = 8,
        down_block_types = (
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        mid_block_type = "UNetMidBlock2DCrossAttn",
        up_block_types = (
            "CrossAttnUpBlock2D", 
            "CrossAttnUpBlock2D", 
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        only_cross_attention = False,
        block_out_channels = args.block_out_channels,
        layers_per_block = 2,
        cross_attention_dim = 768,
        class_embed_type = 'simple_projection',
        projection_class_embeddings_input_dim = 512,
        class_embeddings_concat = True,
    )

    model = UNetWrapper(unet, text_encoder)

    if args.load_from_ckpt_path is not None:
        ckpt_path = args.load_from_ckpt_path
        def load_ckpt(model, ckpt_path, file_name):
            ckpt = torch.load(os.path.join(ckpt_path, file_name), map_location="cpu")
            msg = model.load_state_dict(ckpt, strict=False)
            print(msg)
            return model
        model = load_ckpt(model, ckpt_path, "pytorch_model.bin")

    csv_paths = ['cv_csv_path1', 'cv_csv_path2', 'as_speech_en_csv_path', 'voxceleb_csv_path']
    noise_csv_paths = ['noise_demand_csv_path', 'as_noise_csv_path']

    speech_csvs = [pd.read_csv(getattr(args, path)) for path in csv_paths if hasattr(args, path) and getattr(args, path)]
    noise_csvs = [pd.read_csv(getattr(args, path)) for path in noise_csv_paths if hasattr(args, path) and getattr(args, path)]

    df = pd.concat(speech_csvs, ignore_index=True)
    df_noise = pd.concat(noise_csvs, ignore_index=True)

    train_dataset = AudioDataset(args, df, df_noise, clap_processor)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=CollateFn(text_processor),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    vae.to(accelerator.device, dtype=weight_dtype)
    clap_model.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
                
            text_embed = batch['text_tokens']
            text_embed_mask = batch['text_tokens'].attention_mask

            with torch.no_grad():
                audio_outputs = clap_model.audio_model(
                    input_features=batch['clap_input_features'].to(weight_dtype),
                    is_longer=batch['clap_is_longer'],
                )
                audio_embeds = audio_outputs.pooler_output
                c = clap_model.audio_projection(audio_embeds)

                # randomly replace with unconditional embeddings
                c[torch.rand(c.size(0)) < args.uncond_desc_prob] = uncond_embed.to(c.device, dtype=weight_dtype)
                
            with accelerator.accumulate(model):
                latents = vae.encode(batch['fbank'].unsqueeze(1).to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)

                bsz = latents.shape[0]

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise and compute loss
                noise_pred = model(
                    noisy_latents, 
                    timesteps, 
                    text_embed,
                    text_embed_mask,
                    c,
                    training=True,
                )

                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0
            
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")
                        
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

if __name__ == "__main__":
    main()
