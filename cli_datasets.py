import multiprocessing as mp

import torch
from datasets import load_dataset


def wrap_tokenizer(tokenizer, x, ctx_size):
    return tokenizer(x, return_tensors='pt', truncation=True, padding=True, max_length=ctx_size)


def sample_rp1t(tokenizer, size=128, ctx_size=2048, nproc=1):
    dataset = load_dataset('togethercomputer/RedPajama-Data-1T-Sample', split='train', trust_remote_code=True)
    devset = torch.zeros((size, ctx_size), dtype=torch.int64)
    saved = 0
    print(f'sampling {size} sequences, each with length {ctx_size}, {nproc} processes')
    if nproc > 1:
        with mp.Pool(nproc) as p:
            while saved < size:
                seqs = [(tokenizer, dataset[torch.randint(len(dataset), (size,))]['text'], ctx_size) for _ in range(nproc)]
                tokens = p.starmap(wrap_tokenizer, seqs)
                for i in range(len(tokens)):
                    lens = tokens[i].attention_mask.sum(dim=-1)
                    good = torch.where(lens == ctx_size)[0]
                    if len(good) > 0:
                        if saved + len(good) > size:
                            good = good[:size - saved]
                        devset[saved:saved + len(good)] = tokens[i].input_ids[good]
                        saved += len(good)
                        print(f'selected {saved} sequences')
                del tokens
                torch.cuda.empty_cache()
    else:
        while saved < size:
            tokens = tokenizer(
                dataset[torch.randint(len(dataset), (size,))]['text'],
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=ctx_size
            )
            lens = tokens.attention_mask.sum(dim=-1)
            good = torch.where(lens == ctx_size)[0]
            if len(good) > 0:
                if saved + len(good) > size:
                    good = good[:size - saved]
                devset[saved:saved + len(good)] = tokens.input_ids[good]
                saved += len(good)
                print(f'selected {saved} sequences, {good.shape} good')
    return devset