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
from transformers import MllamaForConditionalGeneration, AutoProcessor

from accelerate import Accelerator
from accelerate import dispatch_model

import pickle

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--base_model', default='meta-llama/Llama-3.2-90B-Vision-Instruct', type=str)
parser.add_argument('--save_path', default='Hessians-Llama_32-90B-Vision-Instruct', type=str)
parser.add_argument('--sample_proc', default=4, type=int)
parser.add_argument('--save_mem', default=False, type=bool)
parser.add_argument('--image_dir', default='/home/aiscuser/yangwang/omnicorpus_samples/images', type=str)
parser.add_argument('--text_dir', default='/home/aiscuser/yangwang/omnicorpus_samples/texts', type=str)
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


# def forward_layer(layer, position_ids, attention_mask, bs, device, in_q, out_q):
#     torch.set_grad_enabled(False)
#     layer = layer.to(device)
#     position_ids = position_ids.to(device)
#     attention_mask = attention_mask.to(device)
# 
#     # Register hooks for all Linear layers in the current transformer layer
#     hooks = []
#     for name, module in layer.named_modules():
#         if isinstance(module, torch.nn.Linear):
#             hook, layer_name = register_H_hook(module, device, f"{layer.name}.{name}")
#             hooks.append(hook)
# 
#     while True:
#         dev_emb = in_q.get()
#         if dev_emb is None:
#             layer = layer.cpu()
#             position_ids = position_ids.cpu()
#             attention_mask = attention_mask.cpu()
#             results = {hook.__name__: hook() for hook in hooks}
#             out_q.put(results)
#             return
# 
#         assert len(dev_emb) % bs == 0
#         for i in range(len(dev_emb) // bs):
#             batch = dev_emb[i * bs:(i + 1) * bs].to(device)
#             with torch.cuda.stream(torch.cuda.Stream()):
#                 output = layer(
#                     batch,
#                     position_ids=position_ids,
#                     attention_mask=attention_mask,
#                     use_cache=False,
#                     output_attentions=False
#                 )[0]
#                 dev_emb[i:i + bs] = output.cpu()
#                 del output
# 
#             # Clear cache every 4 batches
#             if i % (bs * 4) == 0:
#                 torch.cuda.empty_cache()
#                 

# def accumulate(in_q, move_q, ngpus, args, transformer_layer_index):
#     Hs = {}
#     mus = {}
#     cts = {}
# 
#     for i in range(ngpus):
#         out = in_q.get()
#         if i == 0:
#             for key in out:
#                 Hs[key] = torch.zeros(out[key][0].shape, dtype=out[key][0].dtype)
#                 mus[key] = torch.zeros(out[key][1].shape, dtype=out[key][1].dtype)
#                 cts[key] = 0
#         for key in out:
#             Hs[key].add_(out[key][0])
#             mus[key].add_(out[key][1])
#             cts[key] += out[key][2]
# 
#     # keys = list(Hs.keys())
# 
#     for key in Hs:
#         mus[key].div_(cts[key])
#         Hs[key].div_(cts[key])
#         Hs[key].addmm_(-mus[key].unsqueeze(-1), mus[key].unsqueeze(0))
#         save_path = f"{args.scratch_path}/{transformer_layer_index}_{key}.pt" if args.scratch_path is not None else f"{args.save_path}/{transformer_layer_index}_{key}.pt"
#         torch.save({
#             'flatH': sym_to_flat(Hs[key].to(torch.float32)),
#             'mu': mus[key].to(torch.float32),
#             'n': Hs[key].shape[0],
#             'ct': cts[key]
#         }, save_path)
#         if args.scratch_path is not None:
#             move_q.put((
#                 f"{args.scratch_path}/{transformer_layer_index}_{key}.pt",
#                 f"{args.save_path}/{transformer_layer_index}_{key}.pt"
#             ))
# 
#     del Hs, mus, cts, out

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
    model = MllamaForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        use_flash_attention_2=False,
        device_map="auto",
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
            image_files = pickle.load(f)
        print(f'load file list to {args.load_file_list}')
    else: 
        image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        # shuffle
        random.shuffle(image_files)
        image_files = image_files[:args.max_samples]
    
        if args.store_file_list is not None:
            with open(args.store_file_list, 'wb') as f:
                pickle.dump(image_files, f)
            print(f'store file list to {args.store_file_list}')
    
    processor = AutoProcessor.from_pretrained(args.base_model)

    idx = 0
    for image_file in image_files:
        # Get corresponding text file name (assuming same base name, different extension)
        base_name = os.path.splitext(image_file)[0]
        text_file = os.path.join(args.text_dir, f"{base_name}.txt")
        
        if not os.path.exists(text_file):
            print(f"Warning: No matching text file for {image_file}, skipping...")
            continue
            
        # Load image and text
        image = Image.open(os.path.join(args.image_dir, image_file))
        with open(text_file, "r") as f:
            text = f.read()
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": text}
            ]}
        ]
        
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True,
                                                   max_length=8192, truncation=True)
        device = accelerator.device
        
        try:
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(device)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30)
            print(f"index: {idx}, processing {image_file}:")
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

