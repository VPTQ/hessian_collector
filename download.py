# download the model from huggingface

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--threads', default=8, type=int)
args = parser.parse_args()

from huggingface_hub import snapshot_download

snapshot_download(args.model, max_workers=args.threads)
