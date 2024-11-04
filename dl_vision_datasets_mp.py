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

import multiprocessing as mp
from functools import partial
import itertools

def clean_image_url(url):
    """Remove size parameters from image URL"""
    if url is None:
        return None
    # Split URL at '?' and take the base URL
    base_url = url.split('?')[0]
    return base_url

def setup_directories(base_dir):
    # Create directories for text, images, and metadata
    dirs = ['texts', 'images', 'metadata']
    paths = {}
    for dir_name in dirs:
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        paths[dir_name] = dir_path
    return paths


def download_image(images, save_path, retry=5, initial_delay=1):
    url = next((url for url in images if url is not None), images) if isinstance(images, list) else images
    url = clean_image_url(url)
    print(url)
    
    for attempt in range(retry):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                img_format = img.format.lower() if img.format else 'jpeg'
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                ext = f'.{img_format}'
                img.save(save_path + ext, format=img_format)
                return True
                
        except requests.exceptions.RequestException as e:
            delay = initial_delay
            print(f"Attempt {attempt + 1}/{retry} failed: {e}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            continue
            
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}/{retry}: {e}")
            if attempt < retry - 1:
                delay = initial_delay
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


def process_sample(example, output_dir, tokenizer, min_text_len, idx):
    """Process a single sample with image download"""
    text = get_text(example)
    text_len = len(tokenizer.encode(text))
    
    if text_len > min_text_len:
        print(f'Processing sample {idx}, text_len: {text_len}')
        if download_image(example['images'], os.path.join(output_dir, 'images', f'{idx}'), retry=1, initial_delay=1):
            # Save text and metadata
            with open(os.path.join(output_dir, 'texts', f'{idx}.txt'), 'w') as f:
                f.write(text)
            with open(os.path.join(output_dir, 'metadata', f'{idx}.json'), 'w') as f:
                json.dump(example, f)
            return True
    return False


def main():
    # Configuration
    output_dir = "omnicorpus_samples"
    
    parser = argparse.ArgumentParser(description='Download OmniCorpus dataset with range control')
    parser.add_argument('--start', type=int, default=0, help='Starting index for download range')
    parser.add_argument('--end', type=int, default=None, help='Ending index for download range (exclusive)')
    parser.add_argument('--min_text_len', type=int, default=1024, help='Minimum text length to process')
    args = parser.parse_args()
     
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct") 
    # Load dataset
    dataset = datasets.load_dataset("OpenGVLab/OmniCorpus-CC-210M", streaming=True)
    train_dataset = dataset['train']

    # Process samples
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'texts'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)

    successful_pairs = 0
    total_processed = 0
    
    idx = 0
     
    for example in train_dataset:
        if idx >= args.start:
            total_processed += 1
            if process_sample(example, output_dir, tokenizer, args.min_text_len, idx):
                successful_pairs += 1
                print(f'Processed {total_processed} samples, {successful_pairs} successful pairs')
        else:
            idx += 1
            continue

if __name__ == "__main__":
    main()
     