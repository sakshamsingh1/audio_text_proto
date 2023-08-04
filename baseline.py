# We use this script to test the model on a single audio clip.

import argparse
from scripts.baseline.fsd_baseline import train_sv_fsd, run_zs_fsd
from scripts.baseline.esc_us8k_baseline import train_sv_us8k_esc50, run_zs_us8k_esc50

def main(args):
    if args.data == "fsd50k":
        if args.train_type == "sv":
            train_sv_fsd(args.model_type)
        elif args.train_type == "zs":
            run_zs_fsd(args.model_type)
    
    elif args.data == "us8k" or args.data == "esc50":
        if args.train_type == "sv":
            train_sv_us8k_esc50(args.data,args.model_type)
        elif args.train_type == "zs":
            run_zs_us8k_esc50(args.data, args.model_type)

    else:
        raise ValueError(f"Invalid data type: {args.data}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="clap", help="audioclip or clap")
    parser.add_argument("--data", type=str, default="esc50", help="esc50, us8k, fsd50k")
    parser.add_argument("--train_type", type=str, default="zs", help="zs, sv")

    args = parser.parse_args()
    main(args)
