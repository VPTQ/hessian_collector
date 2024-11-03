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

import os
import json
from tqdm import tqdm
import argparse
import transformers
import time

def setup_directories(base_dir):
    # Create directories for text, images, and metadata
    dirs = ['texts', 'images', 'metadata']
    paths = {}
    for dir_name in dirs:
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        paths[dir_name] = dir_path
    return paths

def download_image(images, save_path, retry=10, initial_delay=1):
    url = next((url for url in images if url is not None), images) if isinstance(images, list) else images
    print(url) 
    # if '.' in url.split('/')[-1]:
    #     ext = os.path.splitext(url)[1].lower()
    # else:
    ext = '.jpg'
        
    for attempt in range(retry):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                img.save(save_path + ext)
                return True
                
        except requests.exceptions.RequestException as e:
            delay = initial_delay * (2 ** attempt)  # Exponential backoff
            print(f"Attempt {attempt + 1}/{retry} failed: {e}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            continue
            
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}/{retry}: {e}")
            if attempt < retry - 1:  # If not the last attempt
                delay = initial_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            else:
                print(f"Failed to download image after {retry} attempts")
                break
    return False

def get_text(example):
    # Save text
    text = " ".join(txt for txt in example["texts"] if txt is not None)
    return text

def main():
    # Configuration
    output_dir = "omnicorpus_samples"
    num_samples = 2  # Number of samples to collect
    min_text_len = 4096
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
     
    # Load dataset
    dataset = datasets.load_dataset("OpenGVLab/OmniCorpus-CC-210M", streaming=True)
    train_dataset = dataset['train']
    
    # print(f'dataset size: {len(train_dataset)}')

    # Process samples
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'texts'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)

    successful_pairs = 0
    with tqdm(total=num_samples) as pbar:
        for i, example in enumerate(train_dataset):
            if successful_pairs >= num_samples:
                break
            text = get_text(example)
            text_len = len(tokenizer.encode(text))
            print(f'check {i}, text_len: {text_len}')
            
            if text_len > min_text_len:
                print(f'successful pairs {successful_pairs}, find {i}, text_len: {text_len}')
                # download_image(example['images_info'][0]['url'], os.path.join(output_dir, 'images', f'{i}.jpg'))
                if download_image(example['images'], os.path.join(output_dir, 'images', f'{i}'), retry=10, initial_delay=1):
                    with open(os.path.join(output_dir, 'texts', f'{i}.txt'), 'w') as f:
                        f.write(text)
                    with open(os.path.join(output_dir, 'metadata', f'{i}.json'), 'w') as f:
                        json.dump(example, f)
                    successful_pairs += 1
            
if __name__ == "__main__":
    main()
     