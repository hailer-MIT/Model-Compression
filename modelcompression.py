"""
modelcompression.py

Refactored, modular, local-friendly script for model compression experiments.

This file replaces the original Colab-oriented notebook-style script and provides:
- data generation
- model definitions (teacher / student)
- training, evaluation utilities
- pruning and quantization helpers
- knowledge distillation
- results saving and plotting

Usage examples:
    python modelcompression.py --action all
    python modelcompression.py --action train --epochs 20

"""

import os
import time
import argparse
from typing import Tuple

import numpy as np
import random
import statistics
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.utils.prune as prune

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_data(n_samples=10000, n_features=20, test_size=0.2, seed=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2, random_state=seed)
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
    return train_loader, test_loader


class CreditRiskModel(nn.Module):
    def __init__(self, input_size=20):
        super().__init__()
        # Upgraded to larger architecture (was CreditRiskModelBig)
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class SmallStudentModel(nn.Module):
    def __init__(self, input_size=20):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





def train_model(model: nn.Module, train_loader: DataLoader, epochs=10, lr=1e-3, device='cpu', weight_decay=1e-5) -> nn.Module:
    """Train model with Adam + optional weight decay and ReduceLROnPlateau scheduler."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        if (epoch + 1) % max(1, epochs // 3) == 0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")
    return model


def evaluate_model(model: nn.Module, test_loader: DataLoader, device='cpu') -> float:
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return 100.0 * correct / total


def measure_inference_time(model: nn.Module, test_loader: DataLoader, device='cpu', repeats=3, num_runs=5) -> float:
    """Measure ms per sample using multiple repeats and torch.inference_mode for low noise.

    Returns the median ms/sample over `repeats` batches of `num_runs` runs.
    """
    model.to(device)
    model.eval()
    # Warm-up
    with torch.inference_mode():
        for xb, _ in test_loader:
            _ = model(xb.to(device))
            break

    def one_run():
        start = time.perf_counter()
        samples = 0
        with torch.inference_mode():
            for xb, _ in test_loader:
                _ = model(xb.to(device))
                samples += xb.size(0)
        end = time.perf_counter()
        return (end - start) * 1000.0 / samples

    trial_times = []
    for _ in range(repeats):
        # average several runs to smooth jitter
        run_vals = [one_run() for _ in range(num_runs)]
        trial_times.append(statistics.mean(run_vals))

    return statistics.median(trial_times)


def get_model_size_mb(model: nn.Module) -> float:
    tmp = os.path.join(MODELS_DIR, "_tmp_model.pth")
    torch.save(model.state_dict(), tmp)
    size_mb = os.path.getsize(tmp) / (1024 * 1024)
    os.remove(tmp)
    return size_mb


def save_model(model: nn.Module, name: str) -> str:
    path = os.path.join(MODELS_DIR, name)
    torch.save(model.state_dict(), path)
    return path


def apply_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Apply global L1 unstructured pruning across all Linear weights and make pruning permanent.

    Note: unstructured pruning may not speed up dense inference unless using sparse kernels.
    """
    pruned = type(model)()  # same class
    pruned.load_state_dict(model.state_dict())
    params_to_prune = []
    for name, module in pruned.named_modules():
        if isinstance(module, nn.Linear):
            params_to_prune.append((module, 'weight'))

    if params_to_prune:
        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        # make pruning permanent
        for module, _ in params_to_prune:
            prune.remove(module, 'weight')

    return pruned


class SparseLinear(nn.Module):
    """Linear layer that stores weight as a sparse COO tensor and computes output via sparse mm."""
    def __init__(self, weight_sparse: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        # weight_sparse: sparse_coo_tensor of shape (out_features, in_features)
        if not weight_sparse.is_sparse:
            raise ValueError("weight_sparse must be a sparse tensor")
        # register sparse weight as buffer so it's saved with state_dict
        self.register_buffer('weight_sparse', weight_sparse.coalesce())
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        # sparse mm: (out, in) @ (in, batch) -> (out, batch)
        out = torch.sparse.mm(self.weight_sparse, x.t()).t()
        if self.bias is not None:
            out = out + self.bias
        return out


def convert_pruned_to_sparse(model: nn.Module) -> nn.Module:
    """Return a copy of model where pruned Linear layers are replaced with SparseLinear.

    This will convert any nn.Linear module into SparseLinear using the current weight tensor (zeros removed).
    """
    # Make a deep copy of the model to avoid modifying original
    import copy
    m = copy.deepcopy(model)

    for name, module in list(m.named_modules()):
        # skip top-level
        if isinstance(module, nn.Linear):
            W = module.weight.detach().clone()
            # get indices of non-zero values
            nz = torch.nonzero(W, as_tuple=False)
            if nz.numel() == 0:
                # all zeros - create very small sparse
                indices = torch.zeros((2, 1), dtype=torch.long)
                values = torch.zeros((1,), dtype=W.dtype)
            else:
                indices = nz.t().contiguous()
                values = W[tuple(indices)]

            weight_sparse = torch.sparse_coo_tensor(indices, values, size=W.size())
            weight_sparse = weight_sparse.coalesce()
            bias = module.bias.detach().clone() if module.bias is not None else None

            # find parent module and attribute name to replace
            parent = m
            parts = name.split('.')
            # if nested, traverse
            if len(parts) > 1:
                parent = m
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                child_name = parts[-1]
            else:
                child_name = name

            # create SparseLinear and set on parent
            sparse_lin = SparseLinear(weight_sparse, bias)
            setattr(parent, child_name, sparse_lin)

    return m


def structured_prune_and_squeeze(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Create a compact model by removing low-importance output neurons from hidden Linear layers.

    This does a simple per-layer neuron selection using L1 norm of each output neuron
    and rebuilds a smaller dense model with chained column/row selection so matrix
    dimensions remain consistent. `amount` is the fraction of neurons to remove
    in each prunable layer (0-1). The final output layer is kept unchanged.
    """
    # Only supports the current sequential-like architecture (fc1..fc5)
    # Extract weights and biases
    state = model.state_dict()

    # Layer names expected in this model
    layer_names = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5']

    # Load original weights
    weights = {}
    biases = {}
    for ln in layer_names:
        w_key = f'{ln}.weight'
        b_key = f'{ln}.bias'
        if w_key in state:
            weights[ln] = state[w_key].clone()
        if b_key in state:
            biases[ln] = state[b_key].clone()

    # We'll keep output size for final layer (fc5) fixed
    keep_indices = {}

    # For fc1..fc4, compute keep indices sequentially
    prev_keep = None
    for ln in ['fc1', 'fc2', 'fc3', 'fc4']:
        W = weights[ln]  # shape (out, in)
        # If previous layer selected subset, restrict input columns
        if prev_keep is not None:
            W = W[:, prev_keep]
        # Compute L1 per output neuron
        norms = W.abs().sum(dim=1)
        out = W.size(0)
        keep_n = max(1, int(out * (1.0 - amount)))
        # keep indices with largest norms
        _, idx = torch.topk(norms, k=keep_n, largest=True)
        idx = idx.sort().values
        keep_indices[ln] = idx
        prev_keep = idx

    # Build new compact model class dynamically with reduced sizes
    # Determine new layer sizes
    in_features = weights['fc1'].size(1)
    fc1_out = keep_indices['fc1'].numel()
    fc2_out = keep_indices['fc2'].numel()
    fc3_out = keep_indices['fc3'].numel()
    fc4_out = keep_indices['fc4'].numel()
    fc5_out = weights['fc5'].size(0)

    class CompactModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_features, fc1_out)
            self.bn1 = nn.BatchNorm1d(fc1_out)
            self.fc2 = nn.Linear(fc1_out, fc2_out)
            self.bn2 = nn.BatchNorm1d(fc2_out)
            self.fc3 = nn.Linear(fc2_out, fc3_out)
            self.fc4 = nn.Linear(fc3_out, fc4_out)
            self.fc5 = nn.Linear(fc4_out, fc5_out)
            self.dropout = nn.Dropout(0.25)

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            x = self.dropout(x)
            x = F.relu(self.fc4(x))
            x = self.fc5(x)
            return x

    compact = CompactModel()

    # Fill compact weights from original using keep indices
    # fc1
    k1 = keep_indices['fc1']
    compact.fc1.weight.data.copy_(weights['fc1'][k1, :])
    if 'fc1' in biases:
        compact.fc1.bias.data.copy_(biases['fc1'][k1])

    # fc2: rows = keep_indices['fc2'], cols = keep_indices['fc1']
    k2 = keep_indices['fc2']
    compact.fc2.weight.data.copy_(weights['fc2'][k2][:, keep_indices['fc1']])
    if 'fc2' in biases:
        compact.fc2.bias.data.copy_(biases['fc2'][k2])

    # fc3
    k3 = keep_indices['fc3']
    compact.fc3.weight.data.copy_(weights['fc3'][k3][:, keep_indices['fc2']])
    if 'fc3' in biases:
        compact.fc3.bias.data.copy_(biases['fc3'][k3])

    # fc4
    k4 = keep_indices['fc4']
    compact.fc4.weight.data.copy_(weights['fc4'][k4][:, keep_indices['fc3']])
    if 'fc4' in biases:
        compact.fc4.bias.data.copy_(biases['fc4'][k4])

    # fc5: rows stay same (2), cols correspond to keep_indices['fc4']
    compact.fc5.weight.data.copy_(weights['fc5'][:, keep_indices['fc4']])
    if 'fc5' in biases:
        compact.fc5.bias.data.copy_(biases['fc5'])

    return compact


def apply_quantization(model: nn.Module) -> nn.Module:
    # dynamic quantization works on CPU
    quantized = type(model)()  # instantiate same class
    quantized.load_state_dict(model.state_dict())
    qmodel = torch.quantization.quantize_dynamic(quantized, {nn.Linear}, dtype=torch.qint8)
    return qmodel


def train_distillation(teacher: nn.Module, train_loader: DataLoader, epochs=10, device='cpu') -> nn.Module:
    teacher.to(device)
    teacher.eval()
    student = SmallStudentModel()
    student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    T = 3.0
    for epoch in range(epochs):
        total_loss = 0.0
        student.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                t_probs = F.softmax(teacher(xb) / T, dim=1)
            s_logits = student(xb)
            s_log_probs = F.log_softmax(s_logits / T, dim=1)
            loss_kd = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (T * T)
            loss_ce = F.cross_entropy(s_logits, yb)
            loss = 0.7 * loss_kd + 0.3 * loss_ce
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % max(1, epochs // 3) == 0:
            print(f"Distill Epoch {epoch+1}/{epochs} - loss: {total_loss/len(train_loader):.4f}")
    return student


def plot_results(names, sizes, times, accuracies, out_path=os.path.join(RESULTS_DIR, 'compression_results.png')):
    # Create a figure with 2 rows: top = 3 bar charts, bottom = table
    fig = plt.figure(constrained_layout=True, figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.bar(names, sizes, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    ax0.set_title('Model Size (MB)')

    ax1.bar(names, times, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    ax1.set_title('Inference Time (ms)')

    ax2.bar(names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    ax2.set_title('Accuracy (%)')

    # Add value labels on top of bars
    def add_labels(ax, values, fmt='{:.3f}', pad_frac=0.02):
        patches = ax.patches
        if len(patches) != len(values):
            # fallback: label by x coord
            for i, val in enumerate(values):
                ax.text(i, val, fmt.format(val), ha='center', va='bottom')
            return
        maxv = max(values) if len(values) > 0 else 1.0
        pad = pad_frac * maxv
        for bar, val in zip(patches, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + pad, fmt.format(val), ha='center', va='bottom')

    add_labels(ax0, sizes, fmt='{:.3f}', pad_frac=0.02)
    add_labels(ax1, times, fmt='{:.4f}', pad_frac=0.02)
    add_labels(ax2, accuracies, fmt='{:.2f}%', pad_frac=0.01)

    # Table row spans all columns
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')

    # Build DataFrame for table
    df = pd.DataFrame({
        'Model': names,
        'Size (MB)': [f"{s:.3f}" for s in sizes],
        'Time (ms)': [f"{t:.4f}" for t in times],
        'Accuracy (%)': [f"{a:.2f}" for a in accuracies]
    })

    table = ax_table.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    plt.suptitle('Model Compression Comparison', fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_all(epochs=10, device='cpu', prune_amount=0.3):
    print('Generating data...')
    train_loader, test_loader = generate_data()

    print('Training baseline model...')
    teacher = CreditRiskModel()
    teacher = train_model(teacher, train_loader, epochs=epochs, device=device)
    save_model(teacher, 'baseline_model.pth')

    baseline_acc = evaluate_model(teacher, test_loader, device=device)
    baseline_size = get_model_size_mb(teacher)
    baseline_time = measure_inference_time(teacher, test_loader, device=device)
    baseline_params = sum(p.numel() for p in teacher.parameters())

    print(f'Baseline acc: {baseline_acc:.2f} | size: {baseline_size:.3f} MB | time: {baseline_time:.3f} ms')

    # Pruning
    pruned = apply_pruning(teacher, amount=prune_amount)
    save_model(pruned, 'pruned_model.pth')
    pruned_acc = evaluate_model(pruned, test_loader, device=device)
    pruned_size = get_model_size_mb(pruned)
    pruned_time = measure_inference_time(pruned, test_loader, device=device)

    # Quantization (CPU only)
    quantized = apply_quantization(teacher)
    save_model(quantized, 'quantized_model.pth')
    quantized_acc = evaluate_model(quantized, test_loader, device='cpu')
    quantized_size = get_model_size_mb(quantized)
    quantized_time = measure_inference_time(quantized, test_loader, device='cpu')

    # Distillation
    student = train_distillation(teacher, train_loader, epochs=epochs, device=device)
    save_model(student, 'distilled_model.pth')
    distilled_acc = evaluate_model(student, test_loader, device=device)
    distilled_size = get_model_size_mb(student)
    distilled_time = measure_inference_time(student, test_loader, device=device)

    names = ['Baseline', 'Pruned', 'Quantized', 'Distilled']
    sizes = [baseline_size, pruned_size, quantized_size, distilled_size]
    times = [baseline_time, pruned_time, quantized_time, distilled_time]
    accs = [baseline_acc, pruned_acc, quantized_acc, distilled_acc]

    plot_path = plot_results(names, sizes, times, accs)
    print('Saved plot to', plot_path)

    # Save a small summary text
    summary = {
        'names': names,
        'size_mb': sizes,
        'time_ms': times,
        'accuracy_pct': accs,
        'baseline_params': baseline_params,
    }
    # Save machine-readable summary: CSV + serialized
    df = pd.DataFrame({
        'model': names,
        'size_mb': sizes,
        'time_ms': times,
        'accuracy_pct': accs
    })
    csv_path = os.path.join(RESULTS_DIR, 'summary.csv')
    df.to_csv(csv_path, index=False)
    torch.save(summary, os.path.join(RESULTS_DIR, 'summary.pt'))
    print('Saved summary to', os.path.join(RESULTS_DIR, 'summary.pt'))
    print('Saved CSV summary to', csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['all', 'train', 'prune', 'quantize', 'distill', 'eval'], default='all')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--prune-amount', type=float, default=0.3, help='Global pruning amount (0-1)')
    # script uses the larger model by default
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if args.action == 'all':
        run_all(epochs=args.epochs, device=args.device, prune_amount=args.prune_amount)
    else:
        # For simplicity, run 'all' for other actions in this minimal refactor
        run_all(epochs=args.epochs, device=args.device, prune_amount=args.prune_amount)


if __name__ == '__main__':
    main()