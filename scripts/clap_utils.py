import torch
import librosa
import numpy as np
import laion_clap

def get_clap_model():
    model = laion_clap.CLAP_Module(enable_fusion=True)
    model.load_ckpt()
    return model