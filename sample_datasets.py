from datasets import load_dataset
from transformers import AutoTokenizer
import os
import json

num_samples = 20000
token_length_threshold = 8192 * 2
model_name = 'meta-llama/Llama-3.1-70B-Instruct'
dataset = load_dataset("codeparrot/github-code-clean", name="all-mit", split="train", keep_in_memory=True, num_proc=95, trust_remote_code=True).shuffle(seed=42)
output_dir = 'sampled_data'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Streamlined function for filtering and sampling
def filter_and_sample(dataset, sample_size, token_length_threshold):
    count = 0
    for example in dataset:
        # print(f'example: {example}')
        tokens = tokenizer(example['code'], max_length=None)
        if len(tokens['input_ids']) >= token_length_threshold:
            print(f"Sampled example {count+1} with length {len(tokens['input_ids'])}")
            yield example
            count += 1
            if count >= sample_size:
                break

# Generating sampled data
filtered_stream = filter_and_sample(dataset, num_samples, token_length_threshold)
sampled_data = list(filtered_stream)

print(f"Number of samples collected: {len(sampled_data)}")

# Save sampled data to files
os.makedirs(output_dir, exist_ok=True)
for idx, sample in enumerate(sampled_data, 1):
    file_path = os.path.join(output_dir, f"{idx}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, ensure_ascii=False, indent=4)
