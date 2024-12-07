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
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import AutoModel, AutoTokenizer

from accelerate import Accelerator
from accelerate import dispatch_model

import pickle
import math

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--base_model', default='OpenGVLab/InternVL2_5-78B', type=str)
parser.add_argument('--save_path', default='Hessians-InternVL2_5-78B', type=str)
parser.add_argument('--sample_proc', default=4, type=int)
parser.add_argument('--save_mem', default=False, type=bool)
parser.add_argument('--image_dir', default='/home/aiscuser/yangwang/omnicorpus_samples/images', type=str)
parser.add_argument('--text_dir', default='/home/aiscuser/yangwang/omnicorpus_samples/texts', type=str)
parser.add_argument('--max_samples', default=10000, type=int)
parser.add_argument('--load_file_list', default=None, type=str)
parser.add_argument('--store_file_list', default=None, type=str)
parser.add_argument('--num_part', default=1, type=int)
parser.add_argument('--part_id', default=0, type=int)
parser.add_argument('--start_idx', default=0, type=int)

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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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

    def split_model(model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            'InternVL2_5-1B': 24, 'InternVL_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
            'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
        # Since the first GPU will be used for ViT, treat it as 0.25 a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.1))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.1)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    device_map = split_model(args.base_model.split('/')[-1])
    model = AutoModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map=device_map).eval()

    model.language_model.model.rotary_emb = model.language_model.model.rotary_emb.to('cuda:0')

    model.tie_weights()
    # device_map = dispatch_model(model, device_map="auto")
    # transformer_layers = distribute_layers_across_gpus(model, num_gpus)
    print("loaded model!")
    model = model.eval()
    model = accelerator.prepare(model)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1, do_sample=False)
    
    hooks = []
    for layer_idx, layer in enumerate(model.named_modules()):
        print(f'layer_idx: {layer_idx}, layer: {layer}')
        if isinstance(layer[1], torch.nn.Linear):
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
        if args.store_file_list is not None:
            with open(args.store_file_list, 'wb') as f:
                pickle.dump(image_files, f)
            print(f'store file list to {args.store_file_list}')


    idx = 0
    for image_file in image_files:
        # Get corresponding text file name (assuming same base name, different extension)
        base_name = os.path.splitext(image_file)[0]
        text_file = os.path.join(args.text_dir, f"{base_name}.txt")
        
        if not os.path.exists(text_file):
            print(f"Warning: No matching text file for {image_file}, skipping...")
            continue
        
        with open(text_file, "r") as f:
            text = f.read()
                
        def _truncate_and_decode(text, tokenizer, max_length):
            # Encode the text with truncation
            encoded = tokenizer.encode(text)    
            encoded = encoded[:max_length]
            # Decode the truncated text
            truncated_text = tokenizer.decode(encoded, skip_special_tokens=True)
            return truncated_text

        text = _truncate_and_decode(text, tokenizer, 8192) 

        device = accelerator.device

        pixel_values = load_image(os.path.join(args.image_dir, image_file), max_num=8).to(torch.bfloat16).to(device)
        # text = tokenizer(text, return_tensors="pt").to(device)
        # print(f'pixel_values: {pixel_values.shape}, text: {text.shape}')
        with torch.no_grad():
            # outputs = model.generate(**inputs, max_new_tokens=1)
            response = model.chat(tokenizer, pixel_values, text, generation_config)
            print(f"index: {idx}, processing {image_file}:")
            # print(processor.decode(outputs[0]))
            print("-" * 50)
          
        # try:
        #     pixel_values = load_image(os.path.join(args.image_dir, image_file), max_num=12).to(torch.bfloat16).to(device)
        #     text = tokenizer(text, return_tensors="pt").to(device)
        #     # print(f'pixel_values: {pixel_values.shape}, text: {text.shape}')
        #     with torch.no_grad():
        #         # outputs = model.generate(**inputs, max_new_tokens=1)
        #         response = model.chat(tokenizer, pixel_values, text, generation_config)
        #         print(f"index: {idx}, processing {image_file}:")
        #         # print(processor.decode(outputs[0]))
        #         print("-" * 50)
        #      
        # except Exception as e:
        #     print(f"Error processing {image_file}: {e}")
        #     continue
        
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

