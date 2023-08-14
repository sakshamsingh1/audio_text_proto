import argparse
import torch
import os
from scripts.audioclip_utils import get_norm_audio_embd, get_audioclip_model
from common_utils import get_clap_model, get_label_map

def get_mean_embd(model):
    embd_path = f'data/demo/mean_embd_tensor_esc50_{model}_zs.pt'
    mean_embd_tensor = torch.load(embd_path)
    return mean_embd_tensor

def get_near_dist_class_names(mean_embd, curr_embd, top_n, index_to_label):
    distances = torch.sum((mean_embd - curr_embd)**2, dim=1).sqrt()
    _, top_indices = torch.topk(distances, top_n, largest=False)
    top_classes = [index_to_label[index] for index in top_indices.tolist()]
    
    return top_classes

def main(args):
    TOP_N = 5 # keep it less the number of classes (for esc50: 1-50 )
    if args.model_type == 'proto-lc':
        args.model_type = 'clap'
    elif args.model_type == 'proto-ac':
        args.model_type = 'audioclip'
    else:
        raise ValueError(f'Unknown model type: {args.model_type}')
        
    if args.model_type == 'audioclip':
        PROMPT = 'This is '
        model = get_audioclip_model()
        audio_embd = get_norm_audio_embd([args.audio_path], model, mono=False)

    elif args.model_type == 'clap':
        PROMPT = 'This is a sound of '
        model = get_clap_model()
        # import pdb; pdb.set_trace()
        audio_embd = model.get_audio_embedding_from_filelist([args.audio_path])   

    mean_embd = get_mean_embd(args.model_type)
    print(f'Loaded mean embd for zero shot prototypical {args.model_type} model for esc data')

    data_type = 'esc50'
    label_map = get_label_map(data_type)

    # run inference
    pred_class = get_near_dist_class_names(mean_embd, audio_embd, TOP_N, label_map)
    print(f'Predicted class for {os.path.basename(args.audio_path)}: ')
    for i in range(len(pred_class)):
        print(f'   {i+1}. {PROMPT}{pred_class[i]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="proto-ac", help="proto-lc or proto-ac")
    parser.add_argument("--audio_path", type=str, default="data/demo/airplane_demo.wav", help="audio path")

    args = parser.parse_args()
    main(args)







