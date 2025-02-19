# from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/quantize_llama/hessian_offline_llama.py

import argparse
import datetime
import gc
import os
import random
import json
import time
import atexit

import numpy
import torch
import torch.multiprocessing as mp

from transformers import AutoTokenizer
from safetensors.torch import load_model

import torch.distributed as dist

from deepseek.model import Transformer, ModelArgs, ColumnParallelLinear, RowParallelLinear, Linear

from cli_datasets import sample_rp1t

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

atexit.register(cleanup)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--ckpt_path', default='DeepSeek-V3-mp4', type=str)
parser.add_argument('--save_path', default='Hessians-DeepSeek-V3', type=str)
parser.add_argument('--sample_proc', default=72, type=int)
parser.add_argument('--save_mem', default=False, type=bool)
parser.add_argument('--max_samples', default=2000, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--config', default='config.json', type=str)
parser.add_argument('--row_only', action='store_true')

def register_H_hook(module, device):
    n = module.in_features
    H = torch.zeros(n, n, dtype=torch.float64, device=device)
    mu = torch.zeros(n, dtype=torch.float64, device=device)
    ct = 0

    def H_hook(module, x):
        nonlocal H, mu, ct, n
        x = x[0].reshape(-1, n).to(torch.float64)
        mu.add_(x.sum(dim=0))
        H.addmm_(x.T, x)
        ct += len(x)

    hook = module.register_forward_pre_hook(H_hook)

    def done():
        nonlocal H, mu, ct, hook
        hook.remove()
        H_cpu = H.cpu()
        mu_cpu = mu.cpu()
        ct_copy = ct
        
        hook = None
        H = None
        mu = None
        
        del hook
        del H
        del mu
        del ct

        clean()
        return H_cpu, mu_cpu, ct_copy

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

def hook_forward_layer(layer, device):
    torch.set_grad_enabled(False)
     
    linear_layers = find_linear_layers(layer, args.row_only)
    hook_list = []
    
    for name, module in linear_layers:
        hook_list.append((name, register_H_hook(module, device)))
    return hook_list

def clean():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# input: w1 = w3
def find_linear_layers(module, row_only):
    linear_layers = []
    if row_only:
        for name, module in module.named_modules():
            if isinstance(module, Linear) \
                or isinstance(module, RowParallelLinear) \
                or isinstance(module, ColumnParallelLinear):
                if ('wo' in name) or (('w2' in name) and ('experts' not in name)):
                    linear_layers.append((name, module))
    else:
        for name, module in module.named_modules():
            if isinstance(module, Linear) \
                or isinstance(module, ColumnParallelLinear):
                linear_layers.append((name, module))
    return linear_layers

def main(args):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)  # Set device for NCCL
    
    # Set CPU threads based on mode
    torch.set_num_threads(8)

    # global print
    
    print(f'world_size: {world_size}, rank: {rank}, local_rank: {local_rank}')
    # if rank != 0:
    #     print = lambda *_, **__: None

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(965)
    
    with open(args.config) as f:
        model_args = ModelArgs(**json.load(f))
    print(f"model_args: {model_args}")
    
    if args.tokenizer_path is None:
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    else:
        print(f'load tokenizer from {args.tokenizer_path}')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    if rank == 0:
        print(f"loading dataset on rank {rank}...")
        devset = sample_rp1t(tokenizer, args.max_samples, args.ctx_size, nproc=args.sample_proc)
        devset = devset.to(f'cuda:{rank}')
        clean()
    else:
        devset = torch.zeros((args.max_samples, args.ctx_size), dtype=torch.int64, device=f'cuda:{rank}')
    
    if world_size > 1:
        dist.broadcast(devset, src=0)
        dist.barrier()  # Synchronize all processes
    
    devset = devset.to('cpu')
    
    print("loading model...")
    start_time = time.time()
    
    with torch.device("cpu"):  # Always load model to CPU first
        print(f"Rank {rank}: Creating model")
        model = Transformer(model_args)
        if not args.dry_run:
            print(f"Rank {rank}: Loading model weights")
            load_model(model, os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
            print(f"Rank {rank}: Model load time: {time.time() - start_time:.2f}s")
        else:
            print(f"Rank {rank}: Dry run model load")
    
    dev_emb = []
    
    model.embed = model.embed.to(f'cuda:{rank}')
    
    for idx in range(devset.shape[0]):
        if rank == 0:
            print(f'Starting to process sample {idx} / {devset.shape[0]}')
        try:
            _tokens = devset[idx].to(f'cuda:{rank}')
            _dev_emb = model.embed(_tokens)
            _dev_emb = _dev_emb.unsqueeze(0)
            dev_emb.append(_dev_emb.to("cpu"))
            del _tokens
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'Rank {rank}: Error processing sample {idx}: {str(e)}')
            raise

    del model.embed
    clean()
    
    if rank == 0:
        print(f"dataset dev_emb size: {len(dev_emb)}")

    num_layers = len(model.layers)
    for transformer_layer_index in range(num_layers):
        if rank == 0:
            print(f'processing layer {transformer_layer_index} / {num_layers}')
        transformer_layer = model.layers[transformer_layer_index]
        
        transformer_layer = transformer_layer.to(f'cuda:{rank}')
        hook_list = hook_forward_layer(transformer_layer, f'cuda:{rank}')
        if world_size > 1:
            dist.barrier()
         
        # inference 
        for idx in range(len(dev_emb)):
            if rank == 0:
                print(f'inference sample {idx} / {len(dev_emb)}')
            _dev_emb = dev_emb[idx]
            _start_pos = 0
            _freqs_cis = model.freqs_cis[_start_pos:_start_pos+_dev_emb.size(1)]
            _mask = torch.full((_dev_emb.size(1), _dev_emb.size(1)), float("-inf"), device="cpu").triu_(1)
            _dev_emb = _dev_emb.to(f'cuda:{rank}')
            _freqs_cis = _freqs_cis.to(f'cuda:{rank}')
            _mask = _mask.to(f'cuda:{rank}')
            _dev_emb = transformer_layer(_dev_emb, _start_pos, _freqs_cis, _mask)
            torch.cuda.synchronize()

            dev_emb[idx] = _dev_emb.to("cpu")
            _dev_emb = None
            _freqs_cis = None
            _mask = None
            
            del _dev_emb
            del _freqs_cis
            del _mask
            clean()
        
        if world_size > 1:
            dist.barrier()
        
        # save hessian
        for name, done in hook_list:
            H, mu, ct = done()
            save_path = f'{args.save_path}/{transformer_layer_index}_{name}.pt'
            print(f'{rank}: {transformer_layer_index}_{name}, H: {H.shape}, mu: {mu.shape}, ct: {ct}')
            
            flatH = sym_to_flat(H.to(torch.float32))
            torch.save({
                'flatH': flatH,
                'mu': mu.to(torch.float32),
                'n': H.shape[0],
                'ct': ct
            }, save_path)
           
            flatH = None
            H = None
            mu = None
            del flatH
            del H
            del mu
            clean()
        
        transformer_layer = None
        hook_list = None
        model.layers[transformer_layer_index] = None
        del transformer_layer
        del hook_list
        # del model.layers[transformer_layer_index]
        clean()

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
