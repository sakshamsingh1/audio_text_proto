import torch
import librosa

from scripts.ref_repo.CLAP.src.laion_clap.clap_module import create_model
from scripts.ref_repo.CLAP.src.laion_clap.training.data import get_audio_features
from scripts.ref_repo.CLAP.src.laion_clap.training.data import int16_to_float32, float32_to_int16

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = 'model_K2C_fusion'
enable_fusion = True
data_truncating = 'fusion' # 'rand_trunc' # if run in unfusion mode

base_path = f'../data/input/pretrained/{model_name}/'
param_path = base_path + 'params.txt'
pretrained = base_path + "checkpoints/epoch_top_0.pt"

def find_params_value(file, key):
    # find value of params in params_file
    with open(file, 'r') as f:
        for line in f:
            if key + ': ' in line:
                return line.split(': ')[1].strip()
    return None

def get_model():
    precision = 'fp32'
    amodel = find_params_value(param_path, 'amodel')
    tmodel = find_params_value(param_path, 'tmodel')
    fusion_type = 'aff_2d'

    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision=precision,
        device=device,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type
    )

    model.to(device)
    return model, model_cfg

def get_audio_embd(audio_paths, model, model_cfg):
    audio_input = []
    for audio_path in audio_paths:
        audio_waveform, sr = librosa.load(audio_path, sr=48000)
        audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        audio_dict = {}

        audio_dict = get_audio_features(
            audio_dict, audio_waveform, 480000,
            data_truncating=data_truncating,
            data_filling='repeatpad',
            audio_cfg=model_cfg['audio_cfg']
        )
        audio_input.append(audio_dict)
    # can send a list to the model, to process many audio tracks in one time (i.e. batch size)
    audio_embed = model.get_audio_embedding(audio_input)
    return audio_embed.detach().cpu()