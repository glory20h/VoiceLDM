import os
import argparse

import torch
import torchaudio

from voiceldm import VoiceLDMPipeline


def parse_args(parser):

    parser.add_argument('--desc_prompt', '-d', type=str, help='description prompt')
    parser.add_argument('--cont_prompt', '-c', type=str, help='content prompt')
    parser.add_argument('--audio_prompt', '-a', type=str, help='path to audio file to be used as audio prompt')

    parser.add_argument("--model_config", type=str, default="m", help="configuration for VoiceLDM. 'm' for VoiceLDM-M and 's' for VoiceLDM-S")
    parser.add_argument("--ckpt_path", type=str, help="checkpoint file path for VoiceLDM")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="directory to save generated audio")
    parser.add_argument("--file_name", type=str, help="filename for the generated audio")

    parser.add_argument('--num_inference_steps', type=int, default=50, help='number of inference steps for DDIM sampling')
    parser.add_argument('--audio_length_in_s', type=float, default=10, help='duration of the audio for generation')
    parser.add_argument('--guidance_scale', type=float, help='guidance weight for single classifier-free guidance')
    parser.add_argument('--desc_guidance_scale', type=float, default=5, required=False, help='desc guidance weight for dual classifier-free guidance')
    parser.add_argument('--cont_guidance_scale', type=float, default=7, required=False, help='cont guidance weight for dual classifier-free guidance')
    
    parser.add_argument("--device", type=str, default="auto", help="device to use for audio generation")
    parser.add_argument('--seed', type=int, help='random seed for generation')

    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = torch.device("cuda:0")
        else:
            args.device = torch.device("cpu")
    elif args.device is not None:
        args.device = torch.device(device)
    else:
        args.device = torch.device("cpu")

    pipe = VoiceLDMPipeline(
        args.model_config,
        args.ckpt_path,
        args.device,
    )

    audio = pipe(**vars(args))

    file_name = args.file_name
    if file_name is None:
        file_name = args.desc_prompt or args.audio_prompt
        file_name = file_name + "-" + args.cont_prompt + ".wav"

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, file_name)
    torchaudio.save(save_path, src=audio, sample_rate=16000)


if __name__ == "__main__":
    main()