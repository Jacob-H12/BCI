"""
LMDA-Net Baseline on BCI Competition IV 2a
纯 LMDA-Net，LOSO 9-fold，与 EEGNet 系列对比
使用相同预处理：EA + 8x增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mne
import random
import gc
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler


# ── 数据加载（复用） ──
def load_subject_data(file_path, tmin=0, tmax=4):
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
    print(f"  {file_path.split('/')[-1]}: {data.shape}")
    return data, labels


def euclidean_alignment(data):
    n_trials, n_ch, n_times = data.shape
    covs = np.array([data[i] @ data[i].T / n_times for i in range(n_trials)])
    R_mean = np.mean(covs, axis=0)
    eigvals, eigvecs = np.linalg.eigh(R_mean)
    eigvals = np.maximum(eigvals, 1e-10)
    R_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return np.array([R_inv_sqrt @ data[i] for i in range(n_trials)])


def augment_data_8x(data, labels):
    n_trials, n_ch, n_times = data.shape
    aug_data, aug_labels = [], []
    for i in range(n_trials):
        trial = data[i]; label = labels[i]; trial_std = np.std(trial)

        aug_data.append(trial + 0.02 * trial_std * np.random.randn(*trial.shape))
        aug_labels.append(label)

        shift = random.choice([3, 5, 8]) * random.choice([-1, 1])
        aug_data.append(np.roll(trial, shift, axis=-1))
        aug_labels.append(label)

        aug_data.append(trial * np.random.uniform(0.85, 1.15))
        aug_labels.append(label)

        ch_drop = trial.copy(); ch_drop[np.random.choice(n_ch, random.randint(2, 3), replace=False)] = 0
        aug_data.append(ch_drop); aug_labels.append(label)

        t_mask = trial.copy(); ml = random.randint(40, 100); ms = random.randint(0, n_times - ml)
        t_mask[:, ms:ms + ml] = 0; aug_data.append(t_mask)
        aug_labels.append(label)

        aug_data.append(np.roll(trial + 0.015 * trial_std * np.random.randn(*trial.shape),
                        random.choice([2, 4, 6]) * random.choice([-1, 1]), axis=-1))
        aug_labels.append(label)

        combo = trial * np.random.uniform(0.9, 1.1); combo[np.random.choice(n_ch, 2, replace=False)] = 0
        aug_data.append(combo); aug_labels.append(label)

    combined_data = np.concatenate([data, np.array(aug_data)])
    combined_labels = np.concatenate([labels, np.array(aug_labels)])
    perm = np.random.permutation(len(combined_data))
    print(f"  增强: {n_trials} -> {len(combined_data)} ({len(combined_data)/n_trials:.1f}x)")
    return combined_data[perm], combined_labels[perm]


# ── LMDA-Net 模型 ──
class EEGDepthAttention(nn.Module):
    def __init__(self, W, C, k=7):
        super().__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)
        return y * self.C * x


class LMDA(nn.Module):
    def __init__(self, chans=22, samples=1001, num_classes=4, depth=9, kernel=75,
                 channel_depth1=24, channel_depth2=9, ave_depth=1, avepool=5):
        super().__init__()
        self.ave_depth = ave_depth
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )

        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )

        # 自动计算 flatten 维度
        out = torch.ones((1, 1, chans, samples))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        N, C, H, W = out.size()
        self.depthAttention = EEGDepthAttention(W, C, k=7)
        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out = out.cpu().data.numpy().shape
        self.feat_dim = n_out[-1] * n_out[-2] * n_out[-3]
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)
        x_time = self.time_conv(x)
        x_time = self.depthAttention(x_time)
        x = self.chanel_conv(x_time)
        x = self.norm(x)
        features = torch.flatten(x, 1)
        cls = self.classifier(features)
        return cls, features


# ── 训练 ──
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            with autocast('cuda'):
                logits, _ = model(X)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train_fold(model, train_loader, val_loader, test_loader, device,
               num_epochs=300, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    scaler = GradScaler('cuda')
    best_val = best_test = 0.0; best_state = None; no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        loss_sum = correct = total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits, _ = model(X)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item()
            correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)
        n_batches = len(train_loader)
        if (epoch + 1) % 20 == 0 or val_acc > best_val:
            print(f"    Ep {epoch+1:03d} | Loss {loss_sum/n_batches:.4f} | "
                  f"Train {correct/total:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc; best_test = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"    ✅ 新最佳 Val={best_val:.4f}, Test={best_test:.4f}")
        else:
            no_improve += 1
        if no_improve >= 50:
            print(f"    🛑 早停 (Ep {epoch+1})"); break

    if best_state: model.load_state_dict(best_state)
    return best_val, best_test


# ── 主流程 ──
if __name__ == "__main__":
    all_files = [f'/mnt/eason_ckp/BCI/BCICIV_2a_gdf/A{i:02d}T_train.gdf' for i in range(1, 10)]
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    sample_data, _ = load_subject_data(all_files[0])
    n_times = sample_data.shape[2]; del sample_data; gc.collect()
    print(f"时间点数: {n_times}")

    # 对比基准
    prev = {
        'EEGNet_CL': [67.0, 42.7, 66.0, 46.2, 28.5, 41.7, 50.3, 68.8, 64.2],
        'EEGNet_TempAttn': [73.6, 38.2, 69.8, 51.0, 33.3, 39.6, 44.8, 73.3, 67.0],
    }

    results = []

    for test_idx in range(9):
        print(f"\n{'='*60}")
        print(f"Fold {test_idx+1}/9 — 测试: A{test_idx+1:02d}")
        print(f"{'='*60}")

        # 加载源被试
        train_data_list, train_label_list = [], []
        for i, f in enumerate(all_files):
            if i == test_idx: continue
            d, l = load_subject_data(f)
            d_aligned = euclidean_alignment(d)
            train_data_list.append(d_aligned)
            train_label_list.append(l)
            del d, d_aligned

        train_data = np.concatenate(train_data_list)
        train_labels = np.concatenate(train_label_list)
        del train_data_list, train_label_list; gc.collect()

        test_data, test_labels = load_subject_data(all_files[test_idx])
        test_data = euclidean_alignment(test_data)

        mean = train_data.mean(axis=(0, 2), keepdims=True)
        std = train_data.std(axis=(0, 2), keepdims=True) + 1e-8
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std

        train_aug, labels_aug = augment_data_8x(train_data, train_labels)
        del train_data, train_labels; gc.collect()

        n = len(train_aug); perm = np.random.permutation(n); n_val = int(n * 0.15)
        X_train = torch.FloatTensor(train_aug[perm[n_val:]]).unsqueeze(1)
        y_train = torch.LongTensor(labels_aug[perm[n_val:]])
        X_val = torch.FloatTensor(train_aug[perm[:n_val]]).unsqueeze(1)
        y_val = torch.LongTensor(labels_aug[perm[:n_val]])
        X_test = torch.FloatTensor(test_data).unsqueeze(1)
        y_test = torch.LongTensor(test_labels)
        del train_aug, labels_aug, test_data, test_labels; gc.collect()

        print(f"  数据: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128,
                                  shuffle=True, pin_memory=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, pin_memory=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, pin_memory=True)

        model = LMDA(chans=22, samples=n_times, num_classes=4, depth=9, kernel=75,
                     channel_depth1=24, channel_depth2=9, avepool=5).to(device)
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

        best_val, best_test = train_fold(
            model, train_loader, val_loader, test_loader, device,
            num_epochs=300, lr=1e-3
        )

        cl = prev['EEGNet_CL'][test_idx]
        ta = prev['EEGNet_TempAttn'][test_idx]
        print(f"  ✅ Fold {test_idx+1} | Test={best_test*100:.1f}% | EEGNet_CL={cl:.1f}% | TempAttn={ta:.1f}%")
        results.append(best_test * 100)

        del model, train_loader, val_loader, test_loader
        del X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect(); torch.cuda.empty_cache()

    # ── 最终结果 ──
    print(f"\n{'='*60}")
    print("LMDA-Net Baseline — LOSO 结果:")
    print(f"{'='*60}")
    print(f"  {'Subj':<6} {'LMDA':>9} {'EEGNet_CL':>11} {'TempAttn':>10}")
    print(f"  {'-'*40}")
    for i in range(9):
        subj = f"A{i+1:02d}"
        acc = results[i]; cl = prev['EEGNet_CL'][i]; ta = prev['EEGNet_TempAttn'][i]
        print(f"  {subj:<6} {acc:>8.1f}% {cl:>10.1f}% {ta:>9.1f}%")

    mean_acc = np.mean(results); std_acc = np.std(results)
    cl_mean = np.mean(prev['EEGNet_CL']); ta_mean = np.mean(prev['EEGNet_TempAttn'])
    print(f"\n  LMDA:       {mean_acc:.1f}% ± {std_acc:.1f}%")
    print(f"  EEGNet_CL:  {cl_mean:.1f}%")
    print(f"  TempAttn:   {ta_mean:.1f}%")
    print(f"  vs CL:      {mean_acc - cl_mean:+.1f}%")
    print(f"  vs TempAttn:{mean_acc - ta_mean:+.1f}%")
    print(f"{'='*60}")
