#imports
import argparse
import torch

from scripts.embd_extract.get_embd import gen_embd

def main(args):
    gen_embd(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="audioclip", help="clap or audioclip")
    parser.add_argument("--dataset_name", type=str, default="esc50", help="esc50, us8k, fsd50k")
    parser.add_argument("--num_workers", type=int, default=1, help="num_workers")
    args = parser.parse_args()

    ## manual args
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)
