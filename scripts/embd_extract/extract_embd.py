#imports
import argparse

from audio_clip.gen_embd_fsd import gen_fsdk_embd
from audio_clip.gen_embd_esc import gen_esc_embd
from audio_clip.gen_embd_us8k import gen_us8k_embd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="clap", help="clap or audioclip")
    parser.add_argument("--dataset_name", type=str, default="esc-50", help="esc-50, us8k, fsd50k")
    args = parser.parse_args()

    if args.dataset_name == "fsd50k":
        gen_fsdk_embd()
    elif args.dataset_name == "esc-50":
        gen_esc_embd()
    elif args.dataset_name == "us8k":
        gen_us8k_embd()




