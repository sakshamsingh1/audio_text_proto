# We use this script to test the model on a single audio clip.

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="proto-lc", help="proto-lc or proto-ac")
    parser.add_argument("--label_data", type=str, default="esc-50", help="esc-50, us8k, fsd50k")
    parser.add_argument("--audio_path", type=str, help="audio path", required=True)

    args = parser.parse_args()







