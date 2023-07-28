from scripts.clap_utils import get_clap_model
import torch
from tqdm import tqdm
from common_utils import Fold_proto, get_label_map, run_inference_fsdk, run_inference


def labels_2_meanEmbd(label_map, obj, topn=35, prompt=''):
    model = get_clap_model()

    labels = []
    for i in range(len(label_map)):
        labels.append(label_map[i])

    text_data = [f'{prompt}{label}' for label in labels]
    text_features = model.get_text_embedding(text_data).detach().cpu()
    audio_features = obj.train_norm_feat

    logits_audio_text = (audio_features @ text_features.t()).detach().cpu()

    class_embd = {}
    mean_embd = {}

    for i in range(len(label_map)):
        class_embd[i] = []

    for label_idx in range(len(labels)):
        conf_values, ids = logits_audio_text[:, label_idx].topk(topn)

        for i in ids:
            curr_audio = obj.train_norm_feat[i]
            class_embd[label_idx].append(curr_audio)

    mean_embd_tensor = torch.empty(len(label_map), 512)
    for i in range(len(label_map)):

        my_list = class_embd[i]
        if len(my_list):
            mean_embd[i] = torch.mean(torch.stack(my_list), dim=0)
        else:
            mean_embd[i] = None
        mean_embd_tensor[i, :] = mean_embd[i]
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

    mean_embd_tensor = torch.empty(len(label_map), 512)
    for i in range(len(labels)):
        my_list = class_embd[i]
        if len(my_list):
            mean_embd[i] = torch.mean(torch.stack(my_list), dim=0)
        else:
            mean_embd[i] = None
        mean_embd_tensor[i, :] = mean_embd[i]
    return mean_embd, mean_embd_tensor

def get_proto_lc(data_type, model_type, train_type):
    obj = Fold_proto(data_type, model_type)
    PROMPT = 'This is a sound of '
    label_map = get_label_map(data_type)
    if train_type == 'sv':
        mean_embd, mean_embd_tensor = audioLabels_2_meanEmbd(label_map, obj)
    else:
        mean_embd, mean_embd_tensor = labels_2_meanEmbd(label_map, obj, topn=35, prompt=PROMPT)
    if data_type == 'fsd50k':
        curr_acc = run_inference_fsdk(obj, mean_embd_tensor)
    else:
        curr_acc = run_inference(obj, mean_embd_tensor)
    print(f' Model=proto_ac, train_type={train_type}, data_type={data_type}, acc/mAP={curr_acc}')