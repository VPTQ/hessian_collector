# Collect Hessian matrix from Transformer models

## Environment

```bash
export PATH=/usr/local/cuda-12/bin/:$PATH
conda env create -f environment.yml
```

## Usage

```bash
python collect_hessian.py --model_name_or_path <model_name_or_path> --save_dir <save_dir>
```
