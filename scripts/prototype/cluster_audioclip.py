import sys
from tqdm import tqdm

from fsd_helpers import *

sys.path.append('/scratch/sk8974/experiments/audio_text/audio_text_dhh/scripts/audioClip/fsd50k/supervised_rep/')
from fsd_NN_baseline_helpers import get_map

sys.path.append('/scratch/sk8974/experiments/audio_text/audio_text_dhh/scripts/audioClip/fsd50k/embd_gen/')
from gen_embd_helpers_fsd import get_model, get_text_embd

aclp, _ = get_model()

# our method
def labels_2_meanEmbd(label_map, obj, model=aclp, topn=None, verbose=False, prompt=''):
    labels = []
    for i in range(len(label_map)):
        labels.append(label_map[i])

    text_data = [[f'{prompt}{label}'] for label in labels]

    text_features = get_text_embd(model, text_data)

    audio_features = obj.train_norm_feat
    audio_features = audio_features.to(device)
    scale_audio_text = torch.clamp(model.logit_scale_at.exp(), min=1.0, max=100.0)
    logits_audio_text = scale_audio_text * audio_features @ text_features.T
    logits_audio_text = logits_audio_text.detach().cpu()

    if verbose:
        print('\t\tTextual Label\t\tFilename, Audio (Confidence)', end='\n\n')

    class_embd = {}
    mean_embd = {}

    for i in range(len(label_map)):
        class_embd[i] = []

    for label_idx in range(len(labels)):

        ids, conf_values = None, None

        conf_values, ids = logits_audio_text[:, label_idx].topk(topn)

        for i in ids:
            curr_audio = obj.train_norm_feat[i]
            class_embd[label_idx].append(curr_audio)

        if verbose:
            query = f'{labels[label_idx]:>25s} ->\t\t'
            results = ', '.join([f'{obj.train_true_labels_name[i]} ({v:06.2%})' for v, i in zip(conf_values, ids)])
            print(query + results)

    mean_embd_tensor = torch.empty(len(label_map), 1024)
    for i in range(len(label_map)):

        my_list = class_embd[i]
        if len(my_list):
            mean_embd[i] = torch.mean(torch.stack(my_list), dim=0)
        else:
            mean_embd[i] = None
        mean_embd_tensor[i, :] = mean_embd[i]
    return mean_embd, class_sim_mean, class_sim_median, silk_feat, silk_class, mean_embd_tensor


# baseline: mean embedding of using true labels
def audioLabels_2_meanEmbd(label_map, obj, verbose=False):
    labels = []
    for i in range(len(label_map)):
        labels.append(label_map[i])

    class_embd = {}
    mean_embd = {}

    for i in range(len(label_map)):
        class_embd[i] = []

    silk_feat = []
    silk_class = []

    for label_idx in range(len(labels)):

        ids = []

        for i, gt in enumerate(obj.train_true_labels):
            if label_idx in gt:  # label_name:
                ids.append(i)
                curr_audio = obj.train_norm_feat[i]
                class_embd[label_idx].append(curr_audio)

                silk_feat.append(curr_audio)
                silk_class.append(label_idx)

        if verbose:
            query = f'{labels[label_idx]:>25s} ->\t\t'
            results = ', '.join([f'{obj.train_true_labels_name[i]}' for i in ids])
            print(query + results)

    mean_embd_tensor = torch.empty(len(label_map), 1024)
    for i in range(len(labels)):
        my_list = class_embd[i]
        if len(my_list):
            mean_embd[i] = torch.mean(torch.stack(my_list), dim=0)
        else:
            mean_embd[i] = None
        mean_embd_tensor[i, :] = mean_embd[i]
    return mean_embd, silk_feat, silk_class, mean_embd_tensor


# phase 2
def run_inference(obj, mean_embd_tensor, dist_hist=False, cm=False):
    total_map = 0
    len_test = obj.test_norm_feat.shape[0]

    for idx in tqdm(range(len_test)):
        label_gt = obj.test_true_labels[idx]

        label_oh = torch.zeros(obj.num_class)
        label_oh = label_oh.scatter_(0, torch.tensor(label_gt), 1)

        curr_audio = obj.test_norm_feat[[idx], :]
        # import pdb; pdb.set_trace()

        pred = get_cos_sim(mean_embd_tensor, curr_audio)
        curr_map = get_map(pred, label_oh, use_sig=True)

        total_map += curr_map

    return total_map / len_test
