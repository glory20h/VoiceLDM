# VoiceLDM

This is a repository for the paper, [VoiceLDM: Text-to-Speech with Environmental Context](https://arxiv.org/abs/2309.13664).

VoiceLDM is a text-to-speech model that enables the manipulation of speech generation using a natural language description prompt that includes environmental contextual information.

<a href='https://voiceldm.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs/2309.13664'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15iFqvZL4cBeJcQaoq4j2sjnbcUjeaPMo?usp=sharing)


## 🔧 Installation
```shell
pip install -r requirements.txt
```

## 📖 Usage

- Generate audio with description prompt and content prompt:
```shell
python generate.py --desc_prompt "She is talking in a park." --cont_prompt "Good morning! How are you feeling today?"
```

- Generate audio with audio prompt and content prompt:
```shell
python generate.py --audio_prompt "whispering.wav" --cont_prompt "Good morning! How are you feeling today?"
```

- Text-to-Speech Example:
```shell
python generate.py --desc_prompt "clean speech" --cont_prompt "Good morning! How are you feeling today?" --desc_guidance_scale 1 --cont_guidance_scale 9
```

- Text-to-Audio Example:
```shell
python generate.py --desc_prompt "trumpet" --cont_prompt "_" --desc_guidance_scale 9 --cont_guidance_scale 1
```

Generated audios will be saved at the default output folder `./outputs`.

## 💡 Tips for Better Audio Generation

### Dual Classifier-Free Guidance Matters!

It's crucial to appropriately adjust the weights for dual classifier-free guidance. We find that this adjustment greatly influences the likelihood of obtaining satisfactory results. Here are some key tips:

1. Some weight settings are more effective for different prompts. Experiment with the weights and find the ideal combination that suits the specific use case.

2. Starting with 7 for both `desc_guidance_scale` and `cont_guidance_scale` is a good starting point.

2. If you feel that the generated audio doesn't align well with the provided content prompt, try decreasing the `desc_guidance_scale` and increase the `cont_guidance_scale`.

3. If you feel that the generated audio doesn't align well with the provided description prompt, try decreasing the `cont_guidance_scale` and increase the `desc_guidance_scale`.

## ⚙️ Full List of Options
```console
usage: generate.py [-h] [--desc_prompt DESC_PROMPT] [--cont_prompt CONT_PROMPT] [--audio_prompt AUDIO_PROMPT] [--model_config MODEL_CONFIG] [--ckpt_path CKPT_PATH]
                   [--output_dir OUTPUT_DIR] [--file_name FILE_NAME] [--num_inference_steps NUM_INFERENCE_STEPS] [--audio_length_in_s AUDIO_LENGTH_IN_S]
                   [--guidance_scale GUIDANCE_SCALE] [--desc_guidance_scale DESC_GUIDANCE_SCALE] [--cont_guidance_scale CONT_GUIDANCE_SCALE] [--device DEVICE] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --desc_prompt DESC_PROMPT, -d DESC_PROMPT
                        description prompt
  --cont_prompt CONT_PROMPT, -c CONT_PROMPT
                        content prompt
  --audio_prompt AUDIO_PROMPT, -a AUDIO_PROMPT
                        path to audio file to be used as audio prompt
  --model_config MODEL_CONFIG
                        configuration for VoiceLDM. 'm' for VoiceLDM-M and 's' for VoiceLDM-S
  --ckpt_path CKPT_PATH
                        checkpoint file path for VoiceLDM
  --output_dir OUTPUT_DIR
                        directory to save generated audio
  --file_name FILE_NAME
                        filename for the generated audio
  --num_inference_steps NUM_INFERENCE_STEPS
                        number of inference steps for DDIM sampling
  --audio_length_in_s AUDIO_LENGTH_IN_S
                        duration of the audio for generation
  --guidance_scale GUIDANCE_SCALE
                        guidance weight for single classifier-free guidance
  --desc_guidance_scale DESC_GUIDANCE_SCALE
                        desc guidance weight for dual classifier-free guidance
  --cont_guidance_scale CONT_GUIDANCE_SCALE
                        cont guidance weight for dual classifier-free guidance
  --device DEVICE       device to use for audio generation
  --seed SEED           random seed for generation
```

## 💾 Data

The CSV files for the processed dataset used to train VoiceLDM can be found in [here](https://github.com/glory20h/voiceldm-data). These files include the transcriptions generated using the Whisper model.

### Speech Segments
- `as_speech_en.csv`
- `cv.csv`
- `voxceleb.csv`

### Non-Speech Segments
- `as_noise.csv`
- `noise_demand.csv`

## 🙏 Acknowledgements
This work would not have been possible without the following repositories:

[HuggingFace Diffusers](https://github.com/huggingface/diffusers)

[HuggingFace Transformers](https://github.com/huggingface/transformers)

[AudioLDM](https://github.com/haoheliu/AudioLDM)

[naturalspeech](https://github.com/heatz123/naturalspeech)

[audioldm_eval](https://github.com/haoheliu/audioldm_eval)

[AudioLDM2](https://github.com/haoheliu/AudioLDM2)