import torch
import torchaudio
import numpy as np
from tqdm.auto import tqdm

from transformers import (
    ClapModel,
    ClapProcessor,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from diffusers import (
    AutoencoderKL, 
    DDIMScheduler, 
    UNet2DConditionModel
)
from diffusers.utils import randn_tensor
from huggingface_hub import hf_hub_download

from .modules import UNetWrapper, TextEncoder


class VoiceLDMPipeline():
    def __init__(
        self,
        model_config = None,
        ckpt_path = None,
        device = None,
    ):
        if model_config is None:
            model_config = "m"

        self.noise_scheduler = DDIMScheduler.from_pretrained("cvssp/audioldm-m-full", subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained("cvssp/audioldm-m-full", subfolder="vae").eval()
        self.vocoder = SpeechT5HifiGan.from_pretrained("cvssp/audioldm-m-full", subfolder="vocoder").eval()
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.text_encoder = TextEncoder(SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")).eval()
        self.text_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

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
            block_out_channels = [128, 256, 384, 640] if model_config == "s" else [192, 384, 576, 960],
            layers_per_block = 2,
            cross_attention_dim = 768,
            class_embed_type = 'simple_projection',
            projection_class_embeddings_input_dim = 512,
            class_embeddings_concat = True,
        )

        if ckpt_path is None:
            ckpt_path = hf_hub_download(
                repo_id=f"glory20h/voiceldm-{model_config}",
                filename=f"voiceldm-{model_config}.ckpt"
            )

        # TODO: Get checkpoints
        def load_ckpt(model, ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt)
            return model

        self.model = load_ckpt(UNetWrapper(unet, self.text_encoder), ckpt_path)

        self.device = device
        self.vae.to(device)
        self.vocoder.to(device)
        self.clap_model.to(device)
        self.text_encoder.to(device)
        self.model.to(device)

    def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        shape = (
            batch_size,
            num_channels_latents,
            height // vae_scale_factor,
            self.vocoder.config.model_in_dim // vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
        return mel_spectrogram

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        waveform = waveform.cpu().float()
        return waveform

    def normalize_wav(self, waveform):
        waveform = waveform - torch.mean(waveform)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        return waveform

    def __call__(
        self,
        desc_prompt,
        cont_prompt,
        audio_prompt = None,
        batch_size = 1,
        num_inference_steps = 100,
        audio_length_in_s = 10,
        do_classifier_free_guidance = True,
        guidance_scale = None,
        desc_guidance_scale = None,
        cont_guidance_scale = None,
        device=None,
        seed=None,
        **kwargs,
    ):
        if guidance_scale is None and desc_guidance_scale is None and cont_guidance_scale is None:
            do_classifier_free_guidance = False

        guidance = None
        if guidance_scale is None:
            guidance = "dual"
        else:
            guidance = "single"

        # description condition
        if audio_prompt is None:
            if do_classifier_free_guidance:
                if guidance == "dual":
                    desc_prompt = [desc_prompt] * 2 + [""] * 2
                if guidance == "single":
                    desc_prompt = [desc_prompt] + [""]

            clap_inputs = self.clap_processor(
                text=desc_prompt, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            embeds = self.clap_model.text_model(**clap_inputs).pooler_output
            c_desc = self.clap_model.text_projection(embeds)
        else:
            audio_sample, sr = torchaudio.load(audio_prompt)
            if sr != 48000:
                audio_sample = torchaudio.functional.resample(audio_sample, orig_freq=sr, new_freq=48000)
            audio_sample = audio_sample[0]

            clap_inputs = self.clap_processor(audios=audio_sample, sampling_rate=48000, return_tensors="pt", padding=True).to(self.device)

            embeds = self.clap_model.audio_model(**clap_inputs).pooler_output
            c_desc = self.clap_model.audio_projection(embeds)

            if do_classifier_free_guidance:
                clap_inputs = self.clap_processor(text=[""], return_tensors="pt", padding=True).to(self.device)
                uncond_embeds = self.clap_model.text_model(**clap_inputs).pooler_output
                uc_desc = self.clap_model.text_projection(uncond_embeds)
                
                if guidance == "dual":
                    c_desc = torch.cat((c_desc, c_desc, uc_desc, uc_desc))
                if guidance == "single":
                    c_desc = torch.cat((c_desc, uc_desc))
                
        # content condition
        if do_classifier_free_guidance:
            if guidance == "dual":
                cont_prompt = ([cont_prompt] + ["_"]) * 2
            if guidance == "single":
                cont_prompt = [cont_prompt] + ["_"]

        cont_tokens = self.text_processor(
            text=cont_prompt, 
            padding=True,
            truncation=True,
            max_length=1000,
            return_tensors="pt"
        ).to(self.device)
        cont_embed = self.text_encoder(cont_tokens)
        cont_embed_mask = cont_tokens.attention_mask.unsqueeze(1)

        with torch.no_grad():
            c_cont, cont_embed_mask = self.model.durator(cont_embed, cont_embed_mask)

        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate
        height = int(audio_length_in_s * 1.024 / vocoder_upsample_factor)
        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)

        # prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.noise_scheduler.timesteps

        # prepare latent variables
        num_channels_latents = self.model.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            c_desc.dtype,
            device=device,
            generator=torch.manual_seed(seed) if seed else None,
            latents=None,
        )

        # prepare extra step kwargs
        extra_step_kwargs = {
            'eta': 0.0,
            'generator': torch.manual_seed(seed) if seed else None,
        }

        # denoising loop
        with torch.no_grad():
            num_warmup_steps = len(timesteps) - num_inference_steps * self.noise_scheduler.order
            for i, t in enumerate(tqdm(timesteps)):
                if guidance == "dual":
                    latent_model_input = torch.cat([latents] * 4) if do_classifier_free_guidance else latents
                if guidance == "single":
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.model(
                    latent_model_input, 
                    t, 
                    c_cont,
                    cont_embed_mask,
                    c_desc,
                )

                # perform guidance
                if do_classifier_free_guidance:
                    if guidance == "dual":
                        n1, n2, n3, n4 = noise_pred.chunk(4)
                        noise_pred = n1 + desc_guidance_scale * (n2 - n4) + cont_guidance_scale * (n3 - n4)
                    if guidance == "single":
                        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noise sample x_t -> x_t-1
                latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            mel_spectrogram = self.decode_latents(latents)
            audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
            audio = audio[:, :original_waveform_length]
            audio = self.normalize_wav(audio)

        return audio