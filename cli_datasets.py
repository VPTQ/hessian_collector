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
        p = mp.Pool(nproc)
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
                    print(saved)
    else:
        # while saved < size:
            # tokens = tokenizer(
            #     dataset[torch.randint(len(dataset), (size,))]['text'],
            #     return_tensors='pt',
            #     truncation=True,
            #     padding=True,
            #     max_length=ctx_size
            # )
            # sample = dataset[torch.randint(len(dataset), (1,))]['text']
            # tokens = tokenizer.apply_chat_template([sample], add_generation_prompt=True)
            # lens = len(tokens[0][1])
            # if lens >= ctx_size:
            #     print(f'tokens: {lens}')
            #     devset[saved:saved + ctx_size] = tokens[0][1][:ctx_size]
            #     saved += ctx_size
        _tmp = "Hello, how are you?"
        tokens = tokenizer(
            [_tmp, _tmp, _tmp, _tmp, _tmp, _tmp, _tmp, _tmp, _tmp, _tmp],
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=ctx_size
        )
        print(f'tokens: {tokens.input_ids.shape}')
        for i, t in enumerate(tokens.input_ids):
            devset[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")
        saved += len(tokens)
    return devset
