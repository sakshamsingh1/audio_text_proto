import torch
from scripts.audioclip_utils import get_text_embd
from common_utils import get_embdDim

def labels_2_meanEmbd(model, label_map, obj, topn=35, prompt = '', model_type='audioclip'):
    
    labels = []
    for i in range(len(label_map)):
        labels.append(label_map[i])

    if model_type == 'audioclip':    
        text_data = [[f'{prompt}{label}'] for label in labels]
        text_features = get_text_embd(text_data)
        audio_features = obj.train_norm_feat
        audio_features = audio_features
        scale_audio_text = torch.clamp(model.logit_scale_at.exp(), min=1.0, max=100.0)
        logits_audio_text = scale_audio_text * audio_features @ text_features.T
        logits_audio_text = logits_audio_text.detach().cpu()

    elif model_type == 'clap':
        text_data = [f'{prompt}{label}' for label in labels]
        text_features = model.get_text_embedding(text_data)
        audio_features = obj.train_norm_feat
        logits_audio_text = (audio_features @ torch.tensor(text_features).t()).detach().cpu()

    class_embd = {}
    mean_embd = {}

    for i in range(len(label_map)):
        class_embd[i] = []

    for label_idx in range(len(labels)):
        conf_values, ids = logits_audio_text[:, label_idx].topk(topn)
        for i in ids:
            curr_audio = obj.train_norm_feat[i]
            class_embd[label_idx].append(curr_audio)
    mean_embd_tensor = torch.empty(len(label_map), get_embdDim(model_type))

    for i in range(len(label_map)):
        my_list = class_embd[i]
        if len(my_list):
            mean_embd[i] = torch.mean(torch.stack(my_list) ,dim=0)
        else:
            mean_embd[i] = None
        mean_embd_tensor[i ,:] = mean_embd[i]
    return mean_embd, mean_embd_tensor