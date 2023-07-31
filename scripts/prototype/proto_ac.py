from scripts.audioclip_utils import get_text_embd, get_audioclip_model
import torch
from tqdm import tqdm
import statistics as stats
from common_utils import Fold_proto_fsd, get_label_map, run_inference_fsdk, run_inference, Fold_var_esc_us8k, get_fold_count, get_clap_model, get_embdDim

def labels_2_meanEmbd(label_map, obj, topn=35, prompt = '', model_type='audioclip'):
    
    labels = []
    for i in range(len(label_map)):
        labels.append(label_map[i])

    if model_type == 'audioclip':    
        model = get_audioclip_model()
        text_data = [[f'{prompt}{label}'] for label in labels]
        text_features = get_text_embd(text_data)
        audio_features = obj.train_norm_feat
        audio_features = audio_features
        scale_audio_text = torch.clamp(model.logit_scale_at.exp(), min=1.0, max=100.0)
        logits_audio_text = scale_audio_text * audio_features @ text_features.T
        logits_audio_text = logits_audio_text.detach().cpu()

    elif model_type == 'clap':
        model = get_clap_model()
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

# baseline: mean embedding of using true labels
def audioLabels_2_meanEmbd_fsd(label_map, obj, model_type):
    labels = []
    for i in range(len(label_map)):
        labels.append(label_map[i])

    class_embd = {}
    mean_embd = {}

    for i in range(len(label_map)):
        class_embd[i] = []

    for label_idx in range(len(labels)):
        ids = []

        for i, gt in enumerate(obj.train_true_labels):
            if label_idx in gt:  # label_name:
                ids.append(i)
                curr_audio = obj.train_norm_feat[i]
                class_embd[label_idx].append(curr_audio)

    mean_embd_tensor = torch.empty(len(label_map), get_embdDim(model_type))
    for i in range(len(labels)):
        my_list = class_embd[i]
        if len(my_list):
            mean_embd[i] = torch.mean(torch.stack(my_list), dim=0)
        else:
            mean_embd[i] = None
        mean_embd_tensor[i, :] = mean_embd[i]
    return mean_embd, mean_embd_tensor

def audioLabels_2_meanEmbd_us8k_esc(label_map, obj, model_type):
    labels = []
    for i in range(len(label_map)):
        labels.append(label_map[i])
    
    class_embd = {}
    mean_embd = {}   
    
    for i in range(len(label_map)):
        class_embd[i] = []
    
    for label_idx in range(len(labels)):
        
        ids = []

        for i, gt in enumerate(obj.train_true_labels):
            if gt == label_idx:#label_name:
                ids.append(i)
                curr_audio = obj.train_norm_feat[i]
                class_embd[label_idx].append(curr_audio)
    
    mean_embd_tensor = torch.empty(len(label_map), get_embdDim(model_type))

    for i in range(len(labels)):
        my_list = class_embd[i]
        if len(my_list):
            mean_embd[i] = torch.mean(torch.stack(my_list),dim=0)                
        else:
            mean_embd[i] = None
        mean_embd_tensor[i, :] = mean_embd[i]
    return mean_embd, mean_embd_tensor

def get_proto_ac_fsdk(data_type, model_type, train_type):
    obj = Fold_proto_fsd(data_type, model_type)
    if model_type == 'audioclip':
        PROMPT = 'This is '
    elif model_type == 'clap':
        PROMPT = 'This is a sound of '
    
    label_map = get_label_map(data_type)
    
    if train_type == 'sv':
        mean_embd, mean_embd_tensor = audioLabels_2_meanEmbd_fsd(label_map, obj, model_type)
    else:
        mean_embd, mean_embd_tensor = labels_2_meanEmbd(label_map, obj, topn=35, prompt=PROMPT, model_type=model_type)
    
    curr_acc = run_inference_fsdk(obj, mean_embd_tensor)
    print(f' Model=proto_ac, train_type={train_type}, data_type={data_type}, acc/mAP={curr_acc}')

def get_proto_ac_esc_us8k(data_type, model_type, train_type):
    accs = []
    fold_count = get_fold_count(data_type)
    
    if model_type == 'audioclip':
        PROMPT = 'This is '
    elif model_type == 'clap':
        PROMPT = 'This is a sound of '
    
    for fold in range(1, fold_count+1):
        obj = Fold_var_esc_us8k(data_type, model_type, FOLD=fold)
        label_map = get_label_map(data_type)
        
        if train_type == 'sv':
            mean_embd, mean_embd_tensor = audioLabels_2_meanEmbd_us8k_esc(label_map, obj, model_type)
        else:
            mean_embd, mean_embd_tensor = labels_2_meanEmbd(label_map, obj, topn=35, prompt=PROMPT, model_type=model_type)

        curr_acc = run_inference(obj, mean_embd_tensor)
        print(f' Fold={fold}, acc/mAP={curr_acc}')
        accs.append(curr_acc)
    
    mean_acc = stats.mean(accs)
    print(f' Model=proto_ac, train_type={train_type}, data_type={data_type}, acc/mAP={mean_acc}')


def get_proto_ac(data_type, model_type, train_type):
    print(f' Started running : Model=proto_ac, train_type={train_type}, data_type={data_type}')
    if data_type == 'fsd50k':
        get_proto_ac_fsdk(data_type, model_type, train_type)
    elif data_type == 'esc50' or data_type == 'us8k':
        get_proto_ac_esc_us8k(data_type, model_type, train_type)
