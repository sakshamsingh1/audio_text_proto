import torch
import librosa
import numpy as np
import laion_clap

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def get_clap_model(args):
    check_pt = 'data/input/630k-audioset-fusion-best.pt'
    model = laion_clap.CLAP_Module(enable_fusion=True)
    model.load_ckpt(ckpt=check_pt)
    model = model.to(args.device)
    return model