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

def process_sample(example_and_idx, output_dir, tokenizer, min_text_len):
    """Process a single sample with image download"""
    example, idx = example_and_idx  # Unpack the tuple here
    text = get_text(example)
    text_len = len(tokenizer.encode(text))
    
    if text_len > min_text_len:
        print(f'Processing sample {idx}, text_len: {text_len}')
        if download_image(example['images'], os.path.join(output_dir, 'images', f'{idx}'), retry=10, initial_delay=1):
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
    min_text_len = 4096
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct") 
    # Load dataset
    dataset = datasets.load_dataset("OpenGVLab/OmniCorpus-CC-210M", streaming=True)
    train_dataset = dataset['train']

    # Process samples
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'texts'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)

    num_processes = mp.cpu_count() - 1 
    pool = mp.Pool(processes=num_processes)
    
    successful_pairs = 0
    # 100 for each process
    batch_size = 10
    # 1M total samples
    required_samples = 1000000
    total_processed = 0
    
    try:
        while True and total_processed < required_samples:  # Continue until we run out of data
            # Collect a batch of samples
            batch = list(itertools.islice(train_dataset, batch_size))
            if not batch:  # If no more samples, break
                break
           
            batch_with_indices = [(example, i + total_processed) for i, example in enumerate(batch)]
            
            process_fn = partial(process_sample, 
                               output_dir=output_dir,
                               tokenizer=tokenizer,
                               min_text_len=min_text_len)
            
            results = list(pool.map(process_fn, batch_with_indices))            
            
            successful_pairs += sum(results)
            total_processed += len(batch)
            
            print(f'Processed batch: {total_processed-len(batch)}-{total_processed}, '
                  f'Successful pairs so far: {successful_pairs}')
            
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
    finally:
        pool.close()
        pool.join()
        
    print(f"Finished processing. Total samples processed: {total_processed}")
    print(f"Total successful pairs: {successful_pairs}")


if __name__ == "__main__":
    main()
     