# from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/quantize_llama/hessian_offline_llama.py

# from lib import utils
import argparse
import datetime
import gc
import os
import random

import numpy
import psutil
import torch
import torch.cuda.streams
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import torch
from PIL import Image

from accelerate import Accelerator
from accelerate import dispatch_model

import pickle

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--base_model', default='Qwen/Qwen2.5-Coder-32B-Instruct', type=str)
parser.add_argument('--save_path', default='Hessians-Qwen25-Coder-32B-Instruct', type=str)
parser.add_argument('--sample_proc', default=4, type=int)
parser.add_argument('--save_mem', default=False, type=bool)
parser.add_argument('--text_dir', default='/home/aiscuser/yangwang/data_sample/', type=str)
parser.add_argument('--max_samples', default=10000, type=int)
parser.add_argument('--load_file_list', default=None, type=str)
parser.add_argument('--store_file_list', default=None, type=str)
parser.add_argument('--num_part', default=1, type=int)
parser.add_argument('--part_id', default=0, type=int)

def move_fn(in_q, async_copy_speed):
    # async copy to avoid slow disk
    while True:
        item = in_q.get()
        if item is None:
            return
        src, tgt = item
        if async_copy_speed > 0:
            os.system(f'rsync --bwlimit={async_copy_speed} {src} {tgt}')
        else:
            os.system(f'rsync {src} {tgt}')
        os.system(f'rm {src}')
        print(f'moved {src} to {tgt}')


def register_H_hook(module, device, save_mem):
    compute_device = device
    n = module.in_features
    if save_mem:
        H = torch.zeros(n, n, dtype=torch.float64, device='cpu')
        mu = torch.zeros(n, dtype=torch.float64, device='cpu')
    else:
        H = torch.zeros(n, n, dtype=torch.float64, device=compute_device)
        mu = torch.zeros(n, dtype=torch.float64, device=compute_device)
    ct = 0

    def H_hook(module, x):
        nonlocal H, mu, ct, n
        
        # Handle tuple input - take first element
        if isinstance(x, tuple):
            x = x[0]
        
        # move to compute device
        x = x.to(compute_device)
        H = H.to(compute_device)
        mu = mu.to(compute_device)
        x = x.reshape(-1, n).to(torch.float64)
        mu.add_(x.sum(dim=0))
        H.addmm_(x.T, x)
        ct += len(x)

        del x
        torch.cuda.empty_cache()
        # move back to cpu to save memory
        if save_mem:
            H = H.to('cpu')
            mu = mu.to('cpu')

    hook = module.register_forward_pre_hook(H_hook)

    def done():
        nonlocal H, mu, ct, hook
        hook.remove()
        return H.cpu(), mu.cpu(), ct

    return done


def flat_to_sym(V, N):
    A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
    idxs = torch.tril_indices(N, N, device=V.device)
    A[idxs.unbind()] = V
    A[idxs[1, :], idxs[0, :]] = V
    return A


def sym_to_flat(A):
    N = A.shape[-1]
    idxs = torch.tril_indices(N, N, device=A.device)
    return A[idxs.unbind()]

def clean():
    gc.collect()
    torch.cuda.empty_cache()

def find_linear_layers(model):
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append((name, module))
    return linear_layers


def main(args):
    accelerator = Accelerator()
    local_rank = accelerator.local_process_index
    num_gpus = torch.cuda.device_count()
     
    if accelerator.is_main_process:
        print(f"Total GPUs available: {torch.cuda.device_count()}")

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.tie_weights()
    # device_map = dispatch_model(model, device_map="auto")
    # transformer_layers = distribute_layers_across_gpus(model, num_gpus)
    print("loaded model!")
    model = model.eval()
    model = accelerator.prepare(model)
    hooks = []
    # HACK: split odd/even layers
    for layer_idx, layer in enumerate(model.named_modules()):
        print(f'layer_idx: {layer_idx}, layer: {layer}')
        # language_model.model.layers.0.mlp.down_proj.pt
        parts = layer[0].split('.')
        print(parts)
        try:
            # Try to parse the layer index from parts that typically contain numbers
            model_layer_idx = int(parts[3])
        except (IndexError, ValueError):
            model_layer_idx = 0
            continue
        
        if isinstance(layer[1], torch.nn.Linear) and model_layer_idx % args.num_part == args.part_id:
        # if isinstance(layer[1], torch.nn.Linear) and 'language_model.model' in layer[0]:
            device = next(layer[1].parameters()).device
            hook = register_H_hook(layer[1], device, args.save_mem)
            hooks.append((f"{layer[0]}", hook))
    
    print(f"hooks: {hooks}")
    
    if args.load_file_list is not None:
        with open(args.load_file_list, 'rb') as f:
            text_files = pickle.load(f)
        print(f'load file list to {args.load_file_list}')
    else: 
        text_files = sorted([f for f in os.listdir(args.text_dir) if f.endswith(('.json'))])
        # shuffle
        random.shuffle(text_files)
        text_files = text_files[:args.max_samples]
    
        if args.store_file_list is not None:
            with open(args.store_file_list, 'wb') as f:
                pickle.dump(text_files, f)
            print(f'store file list to {args.store_file_list}')
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
     
    idx = 0
    import json
    for text_file in text_files:
        # Get corresponding text file name (assuming same base name, different extension)
        base_name = os.path.splitext(text_file)[0]
        text_file = os.path.join(args.text_dir, f"{base_name}.json")
        
        with open(text_file, "r") as f:
            json.load(f)
            # text = f.read()
        text = json.load(f)['code']
        # truncate input text
        def _truncate_and_decode(text, tokenizer, max_length):
            # Encode the text with truncation
            encoded = tokenizer.encode(text)    
            encoded = encoded[:max_length]
            # Decode the truncated text
            truncated_text = tokenizer.decode(encoded, skip_special_tokens=True)
            return truncated_text

        text = _truncate_and_decode(text, processor.tokenizer, 8192) 

         
        device = accelerator.device
        
        # print(f"input_text: {input_text}")
        # print(f'len input_text: {len(input_text)}')
         
        try:
            inputs = tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(device)
            
        except Exception as e:
            print(f"Error processing {text_file}: {e}")
            continue
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30)
            print(f"index: {idx}, processing {text_file}:")
            # print(processor.decode(outputs[0]))
            print("-" * 50)
            
        idx += 1
        if idx % 1 == 0:
            print(f"Processed {idx} samples")
            clean()
    
    # save hessians
    clean() 
    os.makedirs(args.save_path, exist_ok=True)
    for hook in hooks:
        H, mu, ct = hook[1]()
        print(f'save hook: {hook[0]}, H: {H.shape}, mu: {mu.shape}, ct: {ct}')

        mu = mu.to('cuda')
        H = H.to('cuda')
        
        mu = mu.div_(ct)
        H = H.div_(ct)
        H = H.addmm_(-mu.unsqueeze(-1), mu.unsqueeze(0))
        
        mu = mu.to('cpu')
        H = H.to('cpu')
       
        save_path = f"{args.save_path}/{hook[0]}.pt"
        torch.save({
            'flatH': sym_to_flat(H.to(torch.float32)),
            'mu': mu.to(torch.float32),
            'n': H.shape[0],
            'ct': ct
        }, save_path)
        
        del H, mu
        clean()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)

