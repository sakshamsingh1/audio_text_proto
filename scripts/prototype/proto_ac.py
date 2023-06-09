from scripts.audioclip_utils import get_text_embd, get_audioclip_model
import torch
from tqdm import tqdm
from common_utils import get_cos_sim, get_map, Fold_proto, get_label_map

def labels_2_meanEmbd(args, label_map, obj, topn=35, prompt = ''):
    model = get_audioclip_model(args)
    labels = []
    for i in range(len(label_map)):
        labels.append(label_map[i])

    text_data = [[f'{prompt}{label}'] for label in labels]

    text_features = get_text_embd(text_data)

    audio_features = obj.train_norm_feat
    audio_features = audio_features.to(args.device)
    scale_audio_text = torch.clamp(model.logit_scale_at.exp(), min=1.0, max=100.0)
    logits_audio_text = scale_audio_text * audio_features @ text_features.T
    logits_audio_text = logits_audio_text.detach().cpu()

    class_embd = {}
    mean_embd = {}

    for i in range(len(label_map)):
        class_embd[i] = []

    for label_idx in range(len(labels)):
        conf_values, ids = logits_audio_text[:, label_idx].topk(topn)
        for i in ids:
            curr_audio = obj.train_norm_feat[i]
            class_embd[label_idx].append(curr_audio)
    mean_embd_tensor = torch.empty(len(label_map), 1024)

    for i in range(len(label_map)):
        my_list = class_embd[i]
        if len(my_list):
            mean_embd[i] = torch.mean(torch.stack(my_list) ,dim=0)
        else:
            mean_embd[i] = None
        mean_embd_tensor[i ,:] = mean_embd[i]
    return mean_embd, mean_embd_tensor


# baseline: mean embedding of using true labels
def audioLabels_2_meanEmbd(label_map, obj):
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

    mean_embd_tensor = torch.empty(len(label_map), 1024)
    for i in range(len(labels)):
        my_list = class_embd[i]
        if len(my_list):
            mean_embd[i] = torch.mean(torch.stack(my_list), dim=0)
        else:
            mean_embd[i] = None
        mean_embd_tensor[i, :] = mean_embd[i]
    return mean_embd, mean_embd_tensor


def run_inference(obj, mean_embd_tensor):
    total_map = 0
    len_test = obj.test_norm_feat.shape[0]

    for idx in tqdm(range(len_test)):
        label_gt = obj.test_true_labels[idx]
        label_oh = torch.zeros(obj.num_class)
        label_oh = label_oh.scatter_(0, torch.tensor(label_gt), 1)
        curr_audio = obj.test_norm_feat[[idx], :]
        pred = get_cos_sim(mean_embd_tensor, curr_audio)
        curr_map = get_map(pred, label_oh, use_sig=True)
        total_map += curr_map
    return total_map / len_test

def get_proto_ac(args, data_type, model_type, train_type):
    obj = Fold_proto(data_type, model_type)
    PROMPT = 'This is a sound of '
    label_map = get_label_map(data_type)
    if train_type == 'sv':
        mean_embd, mean_embd_tensor = audioLabels_2_meanEmbd(label_map, obj)
    else:
        mean_embd, mean_embd_tensor = labels_2_meanEmbd(args, label_map, obj, topn=35, prompt=PROMPT)
    curr_acc = run_inference(obj, mean_embd_tensor)
    print(f' Model=proto_ac, train_type={train_type}, data_type={data_type}, acc/mAP={curr_acc}')