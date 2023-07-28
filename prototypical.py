
import argparse
from scripts.prototype.proto_lc import get_proto_lc
from scripts.prototype.proto_ac import get_proto_ac


def main(args):
    if args.model_type == "proto-lc":
        get_proto_lc(args.data, args.model_type, args.train_type)
    elif args.model_type == "proto-ac":
        get_proto_ac(args.data, args.model_type, args.train_type)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="proto-lc", help="proto-lc or proto-ac")
    parser.add_argument("--data", type=str, default="esc50", help="esc50, us8k, fsd50k")
    parser.add_argument("--train_type", type=str, default="zs", help="zs, sv")

    args = parser.parse_args()
    main(args)


