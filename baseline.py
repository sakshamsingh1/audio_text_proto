# We use this script to test the model on a single audio clip.

import argparse
from scripts.baseline.fsd_baseline import train_sv_fsd, run_zs_fsd
# from scripts.baseline.esc_NN import run_esc_NN
# from scripts.baseline.us8k_NN import run_us8k_NN

def main(args):
    if args.data == "fsd50k":
        if args.train_type == "sv":
            train_sv_fsd(args.model_type)
        elif args.train_type == "zs":
            run_zs_fsd(args.model_type)
    elif args.model_type == "esc":
        run_esc_NN(args.model_type)
    elif args.model_type == "us8k":
        run_us8k_NN(args.model_type)
    else:
        raise ValueError(f"Invalid data type: {args.data}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="audioclip", help="audioclip or clap")
    parser.add_argument("--data", type=str, default="fsd50k", help="esc-50, us8k, fsd50k")
    parser.add_argument("--train_type", type=str, default="zs", help="zs, sv")

    args = parser.parse_args()
    main(args)
