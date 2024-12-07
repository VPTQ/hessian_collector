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
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import torch
from PIL import Image
# from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer, AutoProcessor, image_utils
from transformers import AutoModelForCausalLM

from accelerate import Accelerator
from accelerate import dispatch_model

from cli_datasets import sample_rp1t

import pickle

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--base_model', default='Qwen/Qwen2-57B-A14B-Instruct', type=str)
parser.add_argument('--save_path', default='Hessians-Qwen2-57B-A14B-Instruct', type=str)
parser.add_argument('--sample_proc', default=72, type=int)
parser.add_argument('--save_mem', default=False, type=bool)
parser.add_argument('--max_samples', default=2000, type=int)
parser.add_argument('--ctx_size', default=8192, type=int)
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
    
    print(f"args.base_model: {args.base_model}")
    print(f"loading dataset...")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    devset = sample_rp1t(tokenizer, args.max_samples, args.ctx_size, nproc=args.sample_proc)
    dev_emb = model.model.embed_tokens(devset)
    print(f"loaded dataset! {dev_emb.shape}")

    print("loading model...")
    
    model.tie_weights()
    # device_map = dispatch_model(model, device_map="auto")
    # transformer_layers = distribute_layers_across_gpus(model, num_gpus)
    print("loaded model!")
    model = model.eval()
    model = accelerator.prepare(model)
    
    hooks = []
    for layer_idx, layer in enumerate(model.named_modules()):
        print(f'layer_idx: {layer_idx}, layer: {layer}')
        if isinstance(layer[1], torch.nn.Linear):
            device = next(layer[1].parameters()).device
            hook = register_H_hook(layer[1], device, args.save_mem)
            hooks.append((f"{layer[0]}", hook))
    
    print(f"hooks: {hooks}")
    
    for idx in range(args.max_samples):
        # input_ids = dev_emb[idx]
         
        device = accelerator.device
        input_ids = devset[idx].to(device)  # Use devset instead of dev_emb
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            print(f"index: {idx}, processing {input_ids.shape}:")
            print("-" * 50)
            
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

