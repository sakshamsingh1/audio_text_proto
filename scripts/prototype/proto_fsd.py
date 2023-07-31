import torch

from common_utils import Fold_proto_fsd, get_label_map, run_inference_fsdk, get_clap_model, get_embdDim
from scripts.prototype.proto_utils import labels_2_meanEmbd
from scripts.audioclip_utils import get_audioclip_model

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

def get_proto_fsdk(data_type, model_type, train_type):
    obj = Fold_proto_fsd(data_type, model_type)
    if model_type == 'audioclip':
        PROMPT = 'This is '
        model = get_audioclip_model()

    elif model_type == 'clap':
        PROMPT = 'This is a sound of '
        model = get_clap_model()
    
    label_map = get_label_map(data_type)
    
    if train_type == 'sv':
        mean_embd, mean_embd_tensor = audioLabels_2_meanEmbd_fsd(label_map, obj, model_type)
    else:
        mean_embd, mean_embd_tensor = labels_2_meanEmbd(model, label_map, obj, topn=35, prompt=PROMPT, model_type=model_type)
    
    curr_acc = run_inference_fsdk(obj, mean_embd_tensor)
    print(f' Model=proto_ac, train_type={train_type}, data_type={data_type}, acc/mAP={curr_acc}')
