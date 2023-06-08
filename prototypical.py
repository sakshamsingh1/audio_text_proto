# We use this script to test the model on a single audio clip.

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="proto-lc", help="proto-lc or proto-ac")
    parser.add_argument("--data", type=str, default="esc50", help="esc50, us8k, fsd50k")
    parser.add_argument("--train_type", type=str, default="zs", help="zs, sv")

    args = parser.parse_args()



if __name__ == "__main__":
    main()


