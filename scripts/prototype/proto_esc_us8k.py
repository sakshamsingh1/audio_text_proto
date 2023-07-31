import torch
import statistics as stats

from common_utils import Fold_proto_fsd, get_label_map, run_inference, Fold_var_esc_us8k, get_fold_count, get_clap_model, get_embdDim
from scripts.prototype.proto_utils import labels_2_meanEmbd

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


def get_proto_esc_us8k(data_type, model_type, train_type):
    accs = []
    fold_count = get_fold_count(data_type)
    
    if model_type == 'audioclip':
        PROMPT = 'This is '
        model = get_clap_model()
    elif model_type == 'clap':
        PROMPT = 'This is a sound of '
        model = get_clap_model()
    
    for fold in range(1, fold_count+1):
        obj = Fold_var_esc_us8k(data_type, model_type, FOLD=fold)
        label_map = get_label_map(data_type)
        
        if train_type == 'sv':
            mean_embd, mean_embd_tensor = audioLabels_2_meanEmbd_us8k_esc(label_map, obj, model_type)
        elif train_type == 'zs':
            mean_embd, mean_embd_tensor = labels_2_meanEmbd(model, label_map, obj, topn=35, prompt=PROMPT, model_type=model_type)

        curr_acc = run_inference(obj, mean_embd_tensor)
        print(f' Fold={fold}, acc/mAP={curr_acc}')
        accs.append(curr_acc)
    
    mean_acc = stats.mean(accs)
    print(f' Final score: Model=proto_ac, train_type={train_type}, data_type={data_type}, acc/mAP={mean_acc}')