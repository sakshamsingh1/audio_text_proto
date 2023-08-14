
# This script is used to get the prototypical results for the zero-shot and supervised settings.

import argparse
from scripts.prototype.proto_fsd import get_proto_fsdk
from scripts.prototype.proto_esc_us8k import get_proto_esc_us8k

import warnings
warnings.filterwarnings("ignore")

def main(args):
    
    if args.model_type == "proto-lc":
        args.model_type = "clap"
    elif args.model_type == "proto-ac":
        args.model_type = "audioclip"
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    data_type, model_type, train_type = args.data, args.model_type, args.train_type

    print(f' Started running : Model={args.model_type}, train_type={train_type}, data_type={data_type}')
    if data_type == 'fsd50k':
        get_proto_fsdk(data_type, model_type, train_type)
    elif data_type == 'esc50' or data_type == 'us8k':
        get_proto_esc_us8k(data_type, model_type, train_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="proto-lc", help="proto-lc or proto-ac")
    parser.add_argument("--data", type=str, default="fsd50k", help="esc50, us8k, fsd50k")
    parser.add_argument("--train_type", type=str, default="zs", help="zs, sv")

    args = parser.parse_args()
    main(args)


