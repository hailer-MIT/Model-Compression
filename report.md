Model Compression Report — Credit Risk Prediction
===============================================

Problem statement
-----------------
We train a feed‑forward classifier to predict credit risk (binary) from 20 engineered features on a synthetic dataset (10k samples). The goal is to compress the trained model for edge deployment by reducing disk size and inference latency while preserving accuracy.

Methods (implemented)
```markdown
Model Compression — One Page Report
==================================

Problem statement
-----------------
Load a trained credit‑risk classifier and evaluate how compression techniques (pruning, quantization, distillation) affect model disk size, inference latency, and accuracy for edge deployment.

What I loaded / measured
-------------------------
- Loaded model checkpoint: `models/baseline.pth` when available; otherwise the script trains a baseline and saves it.
- Size: serialized file size on disk (MB).
- Inference: mean per‑sample latency (ms) measured with warmup + multiple repeats under `torch.inference_mode`.
- Accuracy: test set accuracy on a held‑out split.

Implemented methods
-------------------
- Baseline: dense feed‑forward network (teacher).
- Weight pruning: global L1 unstructured pruning and a structured pruning+rebuild (`structured_prune_and_squeeze`) that removes low‑importance neurons and produces a smaller dense model.
- Quantization: PyTorch dynamic quantization (float32 → int8) for linear layers.
- Distillation: teacher → smaller student trained with combined KL (soft targets, T=3) + CE (labels).



Key observations / trade‑offs
-----------------------------
- Size: Distillation and quantization gave the largest storage reductions. Unstructured pruning alone did not reduce file size because zeros remain in dense tensors; use structured pruning or export to sparse/quantized formats to shrink files.
- Latency: Quantization sometimes increases measured Python-side latency (due to conversion overhead) but reduces file size; on-device runtimes (ONNX Runtime, mobile backends) frequently show latency improvements. Structured pruning and sparse kernels are necessary to translate parameter sparsity into real speedups.
- Accuracy: Distillation can preserve or slightly improve accuracy for a much smaller model. Pruning at moderate sparsity (30%) maintained accuracy here; aggressive pruning requires fine‑tuning.

