# Dependencies

## Reference environment

The released code and experiment configurations were validated in the following environment:

| Component | Version |
| --- | --- |
| Python | 3.13.9 |
| PyTorch | 2.9.1+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| cuDNN | 9.10.2 |
| NVIDIA driver | 576.88 |
| GPU | NVIDIA GeForce RTX 5080 |
| NumPy | 2.3.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.7 |
| openpyxl | 3.1.5 |
| tqdm | 4.67.1 |
| fair-esm | 2.0.0 |
| transformers | 4.57.6 |

For the closest reproduction, create a clean environment and install the recorded versions:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==2.3.5 pandas==2.3.3 openpyxl==3.1.5 \
  scikit-learn==1.7.2 matplotlib==3.10.7 tqdm==4.67.1 \
  fair-esm==2.0.0 transformers==4.57.6
```

`fair-esm` is required when residue embeddings are generated from a local ESM-2 `.pt` checkpoint. `transformers` is used only when the selected protein language model is stored in Hugging Face format. Training and evaluation from an existing feature cache do not load either encoder package.

GPU execution is strongly recommended for feature extraction and model training. CPU execution is supported for small checks by setting `DEVICE=cpu` or passing `--device cpu`. Different CUDA kernels, drivers, and AMP implementations can produce small floating-point differences even when the random seed and data split are unchanged.
