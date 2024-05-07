# VoiceLDM

This is a repository for the paper, [VoiceLDM: Text-to-Speech with Environmental Context](https://arxiv.org/abs/2309.13664), ICASSP 2024.

<p align="center">
  <img src="main_figure.png"/>
</p>

VoiceLDM is an extension of text-to-audio models so that it is also capable of generating linguistically intelligible speech.

[2024/05 Update] I have now added the code for training VoiceLDM! Refer to [Training](#-training) for more details.

<a href='https://voiceldm.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs/2309.13664'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15iFqvZL4cBeJcQaoq4j2sjnbcUjeaPMo?usp=sharing)


## 🔧 Installation
```shell
pip install git+https://github.com/glory20h/VoiceLDM.git
cd VoiceLDM
pip install -e .
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

3. If you feel that the generated audio doesn't align well with the provided content prompt, try decreasing the `desc_guidance_scale` and increase the `cont_guidance_scale`.

4. If you feel that the generated audio doesn't align well with the provided description prompt, try decreasing the `cont_guidance_scale` and increase the `desc_guidance_scale`.

## ⚙️ Full List of Options
View the full list of options with the following command:
```console
python generate.py -h
```


## 💾 Data

The CSV files for the processed dataset used to train VoiceLDM can be found in [here](https://github.com/glory20h/voiceldm-data). These files include the transcriptions generated using the Whisper model.

### Speech Segments
- `as_speech_en.csv` (English speech segments from AudioSet)
- `cv1.csv` (English speech segments from CommonVoice 13.0 en, it has been split into two to meet the file size limitations on GitHub.)
- `cv2.csv`
- `voxceleb.csv` (English speech segments from VoxCeleb1)

### Non-Speech Segments
- `as_noise.csv` (Non-speech segments from AudioSet)
- `noise_demand.csv` (Non-speech segments from DEMAND)

## 🧠 Training

If you wish to train the model by yourself, follow these steps:

1. **Configuration Setup (The trickiest part):**
    - Navigate to the `configs` folder to find the necessary configuration files. For example, `VoiceLDM-M.yaml` is used for training the VoiceLDM-M model in the paper.
    - Prepare the CSV files used for training. You can download it [here](https://github.com/glory20h/voiceldm-data).
    - Examine the YAML file and adjust the `"paths"` and `"noise_paths"` to the root path of your dataset. Also, take a look at the CSV files and ensure that the `file_path` in these CSV files match the actual file path names in your dataset.
    - Update the paths for `cv_csv_path1`, `cv_csv_path2`, `as_speech_en_csv_path`, `voxceleb_csv_path`, `as_noise_csv_path`, and `noise_demand_csv_path` in the YAML file. You may optionally leave it blank if you do not wish to use the corresponding csv file and training data.
    - You may also adjust other parameters such as the batch size according to your system's capabilities.

2. **Configure Huggingface Accelerate:**
    - Set up Accelerate by running:
        ```shell
        accelerate config
        ```
        This will allow support of CPU, single GPU, and multi-GPU training.
        Follow the on-screen instructions to configure your hardware settings.
3. **Start Training:**
    - Launch the training process with the following example command:
        ```shell
        accelerate launch train.py --config config/VoiceLDM-M.yaml
        ```
    - Training checkpoints will be automatically saved in the `results` folder.

4. **Running Inference:**
    - Once training is complete, you can perform inference using the trained model by specifying the checkpoint path. For example:
        ```shell
        python generate.py --ckpt_path results/VoiceLDM-M/checkpoints/checkpoint_49/pytorch_model.bin --desc_prompt "She is talking in a park." --cont_prompt "Good morning! How are you feeling today?" 
        ``` 

## 🙏 Acknowledgements
This work would not have been possible without the following repositories:

[HuggingFace Diffusers](https://github.com/huggingface/diffusers)

[HuggingFace Transformers](https://github.com/huggingface/transformers)

[AudioLDM](https://github.com/haoheliu/AudioLDM)

[naturalspeech](https://github.com/heatz123/naturalspeech)

[audioldm_eval](https://github.com/haoheliu/audioldm_eval)

[AudioLDM2](https://github.com/haoheliu/AudioLDM2)