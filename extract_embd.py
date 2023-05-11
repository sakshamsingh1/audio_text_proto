#imports
import argparse
import torch

from scripts.embd_extract.audio_clip.gen_embd_esc import gen_esc_audioclip_embd
from scripts.embd_extract.audio_clip.gen_embd_fsd import gen_fsdk_audioclip_embd
from scripts.embd_extract.audio_clip.gen_embd_us8k import gen_us8k_audioclip_embd

from scripts.embd_extract.clap.gen_embd_esc import gen_esc_clap_embd
from scripts.embd_extract.clap.gen_embd_fsd import gen_fsdk_clap_embd
from scripts.embd_extract.clap.gen_embd_us8k import gen_us8k_clap_embd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="clap", help="clap or audioclip")
    parser.add_argument("--dataset_name", type=str, default="esc-50", help="esc-50, us8k, fsd50k")
    parser.add_argument("--num_workers", type=int, default=3, help="num_workers")
    args = parser.parse_args()

    ## manual args
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model_type == "audioclip":
        if args.dataset_name == "fsd50k":
            gen_fsdk_audioclip_embd(args)
        elif args.dataset_name == "esc-50":
            gen_esc_audioclip_embd(args)
        elif args.dataset_name == "us8k":
            gen_us8k_audioclip_embd(args)
    else:
        if args.dataset_name == "fsd50k":
            gen_fsdk_clap_embd()
        elif args.dataset_name == "esc-50":
            gen_esc_clap_embd(args)
        elif args.dataset_name == "us8k":
            gen_us8k_clap_embd(args)




