import datasets
import base64
from PIL import Image
import io
import os
import json
import hashlib
from tqdm import tqdm
import requests
from transformers import AutoTokenizer
import torch

def setup_directories(base_dir):
    # Create directories for text, images, and metadata
    dirs = ['texts', 'images', 'metadata']
    paths = {}
    for dir_name in dirs:
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        paths[dir_name] = dir_path
    return paths

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            img.save(save_path)
            return True
    except Exception as e:
        print(f"Error downloading image: {e}")
    return False

def main():
    # Configuration
    output_dir = "omnicorpus_samples"
    num_samples = 2  # Number of samples to collect
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama")
     
    # Load dataset
    dataset = datasets.load_dataset("OpenGVLab/OmniCorpus-CC-210M", streaming=True)
    train_dataset = dataset['train']

    # Process samples
    successful_pairs = 0
    with tqdm(total=num_samples) as pbar:
        for i, example in enumerate(train_dataset):
            if i >= num_samples:
                break
            print(f'{i}, {example}')

if __name__ == "__main__":
    main()
     