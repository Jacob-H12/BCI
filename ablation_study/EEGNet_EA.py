"""
EEGNet + EA (Euclidean Alignment) — 在标准 EEGNet baseline 上只加 EA 对齐
对比纯 EEGNet baseline，看 EA 的单独贡献

Baseline: 44.2% ± 16.6%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mne
import os
import gc
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler


# ════════════════════════════════════════════════════════
# 1. 数据加载
# ════════════════════════════════════════════════════════
def load_subject_data(file_path, tmin=0.5, tmax=2.5):
    bad_channels = ['EOG-left', 'EOG-central', 'EOG-right']
    raw = mne.io.read_raw_gdf(file_path, preload=True)
    events, event_dict = mne.events_from_annotations(raw)
    raw.info['bads'] += bad_channels
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
    raw.filter(4., 38., fir_design='firwin', picks=picks)

    mi_events = {}
    for ann_str, evt_id in event_dict.items():
        if ann_str in ('769', '770', '771', '772'):
            mi_events[ann_str] = evt_id

    mi_ids_sorted = sorted(mi_events.values())
    min_mi_id = mi_ids_sorted[0]

    epochs = mne.Epochs(raw, events, mi_events, tmin, tmax, proj=True,
                        picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1] - min_mi_id
    data = epochs.get_data()
    print(f"  {file_path.split('/')[-1]}: {data.shape}, labels={np.unique(labels)}")
    return data, labels


# ════════════════════════════════════════════════════════
# 2. 欧几里得对齐 (EA)
# ════════════════════════════════════════════════════════
def euclidean_alignment(data):
    """对单个被试做 EA，返回对齐后数据"""
    n_trials, n_ch, n_times = data.shape
    covs = np.array([data[i] @ data[i].T / n_times for i in range(n_trials)])
    R_mean = np.mean(covs, axis=0)
    eigvals, eigvecs = np.linalg.eigh(R_mean)
    eigvals = np.maximum(eigvals, 1e-10)
    R_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    aligned = np.array([R_inv_sqrt @ data[i] for i in range(n_trials)])
    return aligned


# ════════════════════════════════════════════════════════
# 3. 标准 EEGNet
# ════════════════════════════════════════════════════════
class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_times=500, n_classes=4,
                 F1=8, D=2, F2=16, kern_length=64, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.pointwise = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            dummy = self._features(dummy)
            flat_dim = dummy.shape[1]

        self.classifier = nn.Linear(flat_dim, n_classes)

    def _features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.separable(x)
        x = self.pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x.flatten(1)

    def forward(self, x):
        return self.classifier(self._features(x))


# ════════════════════════════════════════════════════════
# 4. 训练 & 评估
# ════════════════════════════════════════════════════════
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            with autocast('cuda'):
                logits = model(X)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train_fold(model, train_loader, val_loader, test_loader, device,
               num_epochs=300, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    scaler = GradScaler('cuda')

    best_val = 0.0
    best_test = 0.0
    best_state = None
    patience = 50
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0.0
        correct = total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits = model(X)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)

        if (epoch + 1) % 20 == 0 or val_acc > best_val:
            print(f"    Ep {epoch+1:03d} | Loss {loss_sum/len(train_loader):.4f} | "
                  f"Train {correct/total:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"    ✅ 新最佳 Val={best_val:.4f}, Test={best_test:.4f}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"    🛑 早停 (Ep {epoch+1})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_val, best_test


# ════════════════════════════════════════════════════════
# 5. 主程序：9-fold LOSO + EA
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    all_files = [f'/mnt/eason_ckp/BCI/BCICIV_2a_gdf/A{i:02d}T_train.gdf' for i in range(1, 10)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 探测时间维度
    sample_data, _ = load_subject_data(all_files[0])
    n_times = sample_data.shape[2]
    print(f"时间点数: {n_times}")
    del sample_data; gc.collect()

    # Baseline 对比
    baseline_results = {
        'A01': 56.6, 'A02': 29.2, 'A03': 69.4, 'A04': 36.8, 'A05': 24.7,
        'A06': 28.5, 'A07': 30.2, 'A08': 59.4, 'A09': 63.2
    }

    results = []

    for test_idx in range(9):
        print(f"\n{'='*60}")
        print(f"Fold {test_idx+1}/9 — 测试: A{test_idx+1:02d}")
        print(f"{'='*60}")

        # 加载训练数据，每个被试单独做 EA
        train_data_list, train_label_list = [], []
        for i, f in enumerate(all_files):
            if i == test_idx:
                continue
            d, l = load_subject_data(f)
            d_aligned = euclidean_alignment(d)
            train_data_list.append(d_aligned)
            train_label_list.append(l)
            del d, d_aligned

        train_data = np.concatenate(train_data_list, axis=0)
        train_labels = np.concatenate(train_label_list, axis=0)
        del train_data_list, train_label_list; gc.collect()

        # 测试集也做 EA（用自身数据）
        test_data, test_labels = load_subject_data(all_files[test_idx])
        test_data = euclidean_alignment(test_data)

        # per-channel 标准化（基于训练集）
        mean = train_data.mean(axis=(0, 2), keepdims=True)
        std = train_data.std(axis=(0, 2), keepdims=True) + 1e-8
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std

        # 划分训练/验证
        n = len(train_data)
        perm = np.random.permutation(n)
        n_val = int(n * 0.15)

        X_train = torch.FloatTensor(train_data[perm[n_val:]]).unsqueeze(1)
        y_train = torch.LongTensor(train_labels[perm[n_val:]])
        X_val = torch.FloatTensor(train_data[perm[:n_val]]).unsqueeze(1)
        y_val = torch.LongTensor(train_labels[perm[:n_val]])
        X_test = torch.FloatTensor(test_data).unsqueeze(1)
        y_test = torch.LongTensor(test_labels)
        del train_data, train_labels, test_data, test_labels; gc.collect()

        print(f"  数据: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64,
                                  shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128,
                                shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128,
                                 shuffle=False, num_workers=0, pin_memory=True)

        model = EEGNet(n_channels=22, n_times=n_times, n_classes=4,
                       F1=8, D=2, F2=16, kern_length=64, dropout_rate=0.5).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {total_params:,}")

        best_val, best_test = train_fold(
            model, train_loader, val_loader, test_loader, device,
            num_epochs=300, lr=1e-3
        )

        subj = f"A{test_idx+1:02d}"
        bl = baseline_results[subj]
        diff = best_test * 100 - bl
        print(f"  ✅ Fold {test_idx+1} | Test={best_test*100:.1f}% | Baseline={bl:.1f}% | Δ={diff:+.1f}%")
        results.append(best_test)

        del model, train_loader, val_loader, test_loader, X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()
        torch.cuda.empty_cache()

    # ── 最终结果 ──
    print(f"\n{'='*60}")
    print("EEGNet + EA — LOSO 结果:")
    print(f"{'='*60}")
    print(f"  {'Subject':<8} {'EA':>8} {'Baseline':>10} {'Δ':>8}")
    print(f"  {'-'*36}")
    for i, acc in enumerate(results):
        subj = f"A{i+1:02d}"
        bl = baseline_results[subj]
        diff = acc * 100 - bl
        print(f"  {subj:<8} {acc*100:>7.1f}% {bl:>9.1f}% {diff:>+7.1f}%")

    mean_acc = np.mean(results)
    std_acc = np.std(results)
    bl_mean = np.mean(list(baseline_results.values()))
    print(f"\n  EA 平均:       {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
    print(f"  Baseline 平均: {bl_mean:.1f}% ± 16.6%")
    print(f"  提升:          {mean_acc*100 - bl_mean:+.1f}%")
    print(f"{'='*60}")
