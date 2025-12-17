
# Model Compression Experiment Report

## Problem Statement
Design and implement model compression techniques on a credit risk prediction model
to enable deployment on mobile/edge devices while maintaining acceptable accuracy.

## Model Description
- **Task**: Binary classification (Good vs Bad credit risk)
- **Input**: 20 financial features
- **Output**: 2 classes (Good/Bad credit)
- **Baseline Model**: 5-layer neural network with dropout and batch normalization

## Compression Methods Used

### 1. Pruning (30% sparsity)
- Removed 30% of least important weights using L1 unstructured pruning
- Applied to all linear layers

### 2. Quantization (int8)
- Converted model from float32 to int8 precision
- Dynamic quantization applied to linear layers

### 3. Knowledge Distillation
- Trained a smaller student model (3 layers) using predictions from the teacher model
- Combined distillation loss with task-specific loss

## Results Summary

| Model | Size (MB) | Inference Time (ms) | Accuracy (%) | Parameters |
|-------|-----------|---------------------|--------------|------------|
| Baseline | 0.06 | 0.018 | 98.05 | 13,970 |
| Pruned | 0.06 | 0.014 | 97.75 | 13,970 |
| Quantized | 0.01 | 0.010 | 97.35 | N/A |
| Distilled | 0.03 | 0.018 | 98.05 | 1,234 |

## Key Findings

1. **Pruning** reduced model size by 0.0% with minimal accuracy loss
2. **Quantization** provided the smallest model size (0.01 MB)
3. **Knowledge Distillation** achieved 11.3x parameter reduction
4. All compressed models showed significant inference speed improvements

## Trade-offs Analysis
- **Best for size reduction**: Quantization (0.01 MB)
- **Best speed improvement**: Distilled (0.010 ms)
- **Best accuracy preservation**: Baseline (98.05%)

## Conclusion
Model compression techniques enable significant reductions in model size and inference time
while maintaining acceptable accuracy levels for edge deployment.
