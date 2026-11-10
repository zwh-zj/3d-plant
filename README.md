# 3d-plant

## Environment

The training scripts use a conda environment named `3d-gres`.

| Component | Version |
| --- | --- |
| Python | 3.8 |
| CUDA | 11.3 |
| PyTorch | 1.12.1+cu113 |
| torchvision | 0.13.1+cu113 |
| torchaudio | 0.12.1+cu113 |
| torch-scatter | 2.0.9 |
| spconv-cu113 | 2.3.5 |
| transformers | 4.31.0 |
| gorilla-core | 0.2.7.8 |

`nvcc` and `g++` must be on `PATH`. The CUDA toolkit used to build the extensions is 11.3.

```bash
conda create -n 3d-gres python=3.8 -y
conda activate 3d-gres

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install spconv-cu113==2.3.5
pip install -r requirements.txt
```

Check that this interpreter can see the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The expected output starts with `1.12.1+cu113 True`.

## Build the CUDA extensions

Run the commands below with `3d-gres` activated, from the repository root. `gres_model/lib/setup.py` skips the extension when CUDA is not available.

```bash
cd pointnet2
python setup.py build_ext --inplace
cd ../gres_model/lib
python setup.py build_ext --inplace
```

This writes:

- `pointnet2/_ext*.so`
- `gres_model/lib/pointgroup_ops/pointgroup_ops_ext*.so`

Importing `pointgroup_ops` needs `gres_model/lib` on `PYTHONPATH`. From the repository root:

```bash
export PYTHONPATH="$(pwd):$(pwd)/gres_model/lib${PYTHONPATH:+:$PYTHONPATH}"
```

On WSL, also add the Windows GPU driver library:

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Data

| Dataset | Download |
| --- | --- |
| Crops3D | https://doi.org/10.6084/m9.figshare.27313272 |
| Pheno4D | https://www.ipb.uni-bonn.de/html/projects/Pheno4D/Pheno4D.zip |

The Crops3D DOI opens the Figshare record. Download `Crops3D.zip` from that page. The dataset description and split files are also listed at https://github.com/clawCa/Crops3D.

The Pheno4D page is https://www.ipb.uni-bonn.de/data/pheno4d/. The zip above is the full archive linked from that page.

## Text encoder

The model loads RoBERTa-base from `backbones/roberta-base/` with `local_files_only=True`.

Download the text encoder from https://huggingface.co/roberta-base and place it in that directory. The weight file is:

https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
