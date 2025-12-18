Project Overview
================
This project implements and compares three model compression techniques for deep learning models, demonstrated on a credit risk prediction task. The goal is to reduce model size and inference time while keeping high accuracy for edge deployment.

Key points
- Problem: Binary credit-risk classification with 20 input features.
- Implemented compression: pruning (L1), dynamic quantization (INT8), and knowledge distillation (teacher → student).
- Outputs: trained model files, comparison plot (bar charts + results table), and a CSV summary.

Files produced by this repo
- models/ — saved model checkpoints: baseline_model.pth, pruned_model.pth, quantized_model.pth, distilled_model.pth
- results/ — compression_results.png (charts + table), summary.csv, summary.pt
- modelcompression.py — main script to run the experiments
- requirements.txt — Python dependencies

Clone & run (Windows PowerShell)
1. Clone the repo 

    git clone https://github.com/hailer-MIT/Model-Compression.git
    cd Model-Compression

2. Create & activate a virtual environment:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

3. Install dependencies:

    pip install --upgrade pip
    pip install -r requirements.txt

4. Run the full pipeline (example, 10 epochs):

    python modelcompression.py --action all --epochs 10 --device cpu

Notes
- If you have a CUDA-capable GPU and installed a CUDA-enabled PyTorch, set `--device cuda`.
- Non-`all` actions currently run the full pipeline; feel free to modify the script to run specific steps.

Results and interpretation
- After running, open `results/compression_results.png` to view bar charts and a table comparing model size (MB), inference time (ms), and accuracy (%).
- `results/summary.csv` contains the numeric results you can paste into your 1‑page report.

# Model Compression Methods — Credit Risk Prediction

Overview
--------
This repository demonstrates three model compression techniques (pruning, quantization, and distillation) on a credit‑risk classification task. The objective is to reduce serialized model size and inference latency while maintaining classification accuracy for edge deployment.

Repository layout
-----------------
- `modelcompression.py` — main experiment script (train, compress, evaluate, plot).
- `models/` — saved checkpoints (baseline, pruned, quantized, distilled).
- `results/` — `compression_results.png`, `summary.csv`, `summary.pt`.
- `requirements.txt` — Python package list.

Quickstart (Windows PowerShell)
------------------------------
1. Clone and enter the repo:

```powershell
git clone https://github.com/hailer-MIT/Model-Compression.git
cd Model-Compression
```

2. Create & activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies and run the full pipeline (example):

```powershell
pip install --upgrade pip
pip install -r requirements.txt
python modelcompression.py --action all --epochs 10 --device cpu
```

Compression methods — short explanations
--------------------------------------

Pruning
- What it is: remove (set to zero) less important weights in the network to reduce parameter count and (potentially) runtime.
- How used here: global L1 unstructured pruning removes the lowest‑magnitude weights across linear layers. Because zeros are still stored in dense tensors, unstructured pruning does not always reduce file size or CPU runtime. To obtain practical gains we also implement `structured_prune_and_squeeze`, which removes low‑importance neurons and rebuilds a smaller dense model that reduces parameter count and serialized size.

Quantization
- What it is: lower numeric precision (e.g., float32 → int8) to reduce model storage and accelerate integer‑efficient runtimes.
- How used here: PyTorch dynamic quantization (`torch.quantize_dynamic`) on linear layers. This reduces disk size significantly; latency behavior depends on runtime and backend (Python overhead can mask int8 speedups; ONNX/runtime or platform-specific backends often show clearer improvements).

Distillation
- What it is: train a compact student model to mimic a larger teacher’s softened logits, usually using a temperature parameter and a KL loss combined with label loss. Distillation can produce much smaller models with comparable accuracy.
- How used here: a smaller student network is trained with a combined loss: KL divergence between teacher and student logits (temperature T=3) plus cross‑entropy with labels. The student is faster and much smaller on disk.

What we measure
----------------
- Model size: serialized checkpoint size on disk (MB).
- Inference time: mean per‑sample latency (ms) measured after warmup using `torch.inference_mode` and multiple repeats.
- Accuracy: test set accuracy on a held‑out split.

Interpreting results (tips)
--------------------------
- Unstructured pruning may not reduce serialized size or runtime; prefer structured pruning or convert pruned models to sparse formats and test on runtimes that support sparse kernels.
- Quantization reduces storage and can speed up inference on proper backends; measure on the target platform (ONNX Runtime, mobile runtime) for realistic latency results.
- Distillation is effective when you need a much smaller model with little to no accuracy loss.

Recommended next steps for final submission
------------------------------------------
1. Run experiments with multiple random seeds (3–5) and report mean ± std for size, latency, and accuracy.
2. For pruning speed/size wins: apply `structured_prune_and_squeeze`, fine‑tune the smaller model, and/or export quantized models to ONNX for target runtime benchmarking.
3. Package deliverables: `results/summary.csv`, `results/compression_results.png`, `models/` checkpoints, and this README + `report.md`.

Contact
-------
Author: hailer‑MIT (hailomasegede@gmail.com)
