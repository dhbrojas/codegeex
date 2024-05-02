# CodeGeeX Training and Experiments

Boilerplate repository for small scale code language model training.

## Goals

- [ ] Pre-train a 400M causal GPT model at 500,000 tokens/s on 8xH800 (~40B tokens per day).

## Environment

You are expected to use the official NVIDIA PyTorch image.

```bash
# Download the image.
docker pull nvcr.io/nvidia/pytorch:24.04-py3
# Run a container with /workspace mounted inside.
docker run --detach --name codegeex --gpus all --shm-size 1G --volume /workspace:/workspace:rw nvcr.io/nvidia/pytorch:24.04-py3 tail -f /dev/null
# Launch a shell in the container.
docker exec -it codegeex /bin/bash
```

## Dependencies

The official NVIDIA PyTorch image comes pre-packaged with the following dependencies.

| Name               | Version |
| ------------------ | ------- |
| PyTorch            | 2.3.0   |
| PyTorch Triton     | 3.0.0   |
| Flash Attention    | 2.4.2   |
| Transformer Engine | 1.5.0   |

Amonst others.

We manually configure the following additional dependencies.

> Note: For users in China, make sure to run `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`.

### Hugging Face

```
pip install accelerate transformers safetensors datasets tokenizers
```

### TokenGeeX

```
pip install tokengeex
```

### XFormers

```
pip install ninja
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

For users in China, a copy of the source code of XFormers is provided in the `vendor` directory.

```
pip install ninja
pip install vendor/xformers-0.0.26.post1.tar
```

## Fused RMS Norm & RoPE

The Flash Attention repository provides optimised implementations of RMSNorm and RoPE. They're not available on PyPI so we install them manually. We use Flash Attention v2.5.8 for this.

```bash
# Untar the archive.
mkdir -p build
tar -xvf ./vendor/flash-attention-2.5.8.tar -C ./build
# Build the extensions.
cd ./csrc/rotary && pip install .
cd ../layer_norm && pip install .
```
