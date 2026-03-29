"""
EEGNet + EA + 8x增强 + Center Loss + MMD Loss (V2: 0-4s)
新增 MMD：缩小不同源被试特征分布的差异，学习 domain-invariant 特征
对比: +CL 52.8% | V2无CL 51.9%

损失 = CE + 0.03×CenterLoss + 0.1×MMD
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


def augment_data_8x(data, labels, domain_ids=None):
    """8x增强，同时保留 domain_id"""
    n_trials, n_ch, n_times = data.shape
    aug_data, aug_labels, aug_domains = [], [], []
    for i in range(n_trials):
        trial = data[i]; label = labels[i]; trial_std = np.std(trial)
        did = domain_ids[i] if domain_ids is not None else 0

        aug_data.append(trial + 0.02 * trial_std * np.random.randn(*trial.shape))
        aug_labels.append(label); aug_domains.append(did)

        shift = random.choice([3, 5, 8]) * random.choice([-1, 1])
        aug_data.append(np.roll(trial, shift, axis=-1))
        aug_labels.append(label); aug_domains.append(did)

        aug_data.append(trial * np.random.uniform(0.85, 1.15))
        aug_labels.append(label); aug_domains.append(did)

        ch_drop = trial.copy(); ch_drop[np.random.choice(n_ch, random.randint(2, 3), replace=False)] = 0
        aug_data.append(ch_drop); aug_labels.append(label); aug_domains.append(did)

        t_mask = trial.copy(); ml = random.randint(40, 100); ms = random.randint(0, n_times - ml)
        t_mask[:, ms:ms + ml] = 0; aug_data.append(t_mask)
        aug_labels.append(label); aug_domains.append(did)

        aug_data.append(np.roll(trial + 0.015 * trial_std * np.random.randn(*trial.shape),
                        random.choice([2, 4, 6]) * random.choice([-1, 1]), axis=-1))
        aug_labels.append(label); aug_domains.append(did)

        combo = trial * np.random.uniform(0.9, 1.1); combo[np.random.choice(n_ch, 2, replace=False)] = 0
        aug_data.append(combo); aug_labels.append(label); aug_domains.append(did)

    combined_data = np.concatenate([data, np.array(aug_data)])
    combined_labels = np.concatenate([labels, np.array(aug_labels)])
    combined_domains = np.concatenate([domain_ids, np.array(aug_domains)]) if domain_ids is not None else None
    perm = np.random.permutation(len(combined_data))
    print(f"  增强: {n_trials} -> {len(combined_data)} ({len(combined_data)/n_trials:.1f}x)")
    return combined_data[perm], combined_labels[perm], combined_domains[perm] if combined_domains is not None else None


class TemporalSelfAttention(nn.Module):
    """
    轻量级 temporal self-attention，类似 V2 的 UltraLightSharedTemporalAttention
    输入: (batch, F2, 1, T) — pool2 输出
    在时间维度 T 上做 self-attention，让模型学习时间步之间的依赖关系
    """
    def __init__(self, n_channels, time_steps, n_heads=2, dropout=0.15):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = n_channels // n_heads
        self.scale = self.head_dim ** 0.5

        # QKV 线性变换（沿通道维度）
        self.qkv = nn.Linear(n_channels, 3 * n_channels, bias=False)
        self.out_proj = nn.Linear(n_channels, n_channels, bias=False)
        self.ln = nn.LayerNorm(n_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, F2, 1, T)
        b, c, _, t = x.shape
        x_seq = x.squeeze(2).permute(0, 2, 1)  # (batch, T, F2)

        residual = x_seq

        # QKV
        qkv = self.qkv(x_seq)  # (batch, T, 3*F2)
        qkv = qkv.reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, heads, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, heads, T, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(b, t, c)  # (batch, T, F2)

        out = self.out_proj(out)
        out = self.ln(out + residual)  # 残差连接 + LayerNorm

        # 恢复原始shape
        out = out.permute(0, 2, 1).unsqueeze(2)  # (batch, F2, 1, T)
        return out


class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_times=1000, n_classes=4,
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
        # Temporal Self-Attention
        with torch.no_grad():
            dummy_conv = self._conv_features(torch.zeros(1, 1, n_channels, n_times))
            t_dim = dummy_conv.shape[3]  # 时间维度
        self.temporal_attn = TemporalSelfAttention(F2, t_dim, n_heads=2, dropout=0.15)
        with torch.no_grad():
            dummy = self._features(torch.zeros(1, 1, n_channels, n_times))
            self.feat_dim = dummy.shape[1]
        self.classifier = nn.Linear(self.feat_dim, n_classes)

    def _conv_features(self, x):
        x = F.elu(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))
        x = self.drop1(self.pool1(x))
        x = F.elu(self.bn3(self.pointwise(self.separable(x))))
        x = self.drop2(self.pool2(x))
        return x

    def _features(self, x):
        x = self._conv_features(x)
        x = self.temporal_attn(x)  # temporal self-attention
        return x.flatten(1)

    def forward(self, x):
        feat = self._features(x)
        logits = self.classifier(feat)
        return logits, feat


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_c=0.03):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)
        self.lambda_c = lambda_c

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        return self.lambda_c * F.mse_loss(features, batch_centers)


def gaussian_kernel(x, y, sigma=1.0):
    """高斯核"""
    xx = torch.sum(x * x, dim=1, keepdim=True)
    yy = torch.sum(y * y, dim=1, keepdim=True)
    xy = torch.mm(x, y.t())
    dist = xx + yy.t() - 2 * xy
    return torch.exp(-dist / (2 * sigma ** 2))


def mmd_loss(feat, domain_ids, lambda_mmd=0.1):
    """
    多域 MMD：随机抽两个域，计算它们特征分布的 MMD
    """
    unique_domains = torch.unique(domain_ids)
    if len(unique_domains) < 2:
        return torch.tensor(0.0, device=feat.device)

    # 随机选两个域
    idx = torch.randperm(len(unique_domains))[:2]
    d1, d2 = unique_domains[idx[0]], unique_domains[idx[1]]

    feat1 = feat[domain_ids == d1]
    feat2 = feat[domain_ids == d2]

    if len(feat1) < 2 or len(feat2) < 2:
        return torch.tensor(0.0, device=feat.device)

    # 自适应 sigma
    with torch.no_grad():
        all_feat = torch.cat([feat1, feat2], dim=0)
        dists = torch.cdist(all_feat, all_feat)
        sigma = torch.median(dists[dists > 0]).clamp(min=0.1)

    k_xx = gaussian_kernel(feat1, feat1, sigma)
    k_yy = gaussian_kernel(feat2, feat2, sigma)
    k_xy = gaussian_kernel(feat1, feat2, sigma)

    n1, n2 = feat1.size(0), feat2.size(0)
    mmd = k_xx.sum() / (n1 * n1) + k_yy.sum() / (n2 * n2) - 2 * k_xy.sum() / (n1 * n2)

    return lambda_mmd * mmd.clamp(min=0)


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


def train_fold(model, center_loss, train_loader, val_loader, test_loader, device,
               num_epochs=300, lr=1e-3, lambda_mmd=0.1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr, 'weight_decay': 1e-4},
        {'params': center_loss.parameters(), 'lr': lr * 5, 'weight_decay': 0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    scaler = GradScaler('cuda')
    best_val = best_test = 0.0; best_state = None; no_improve = 0

    for epoch in range(num_epochs):
        model.train(); center_loss.train()
        loss_sum = ce_sum = cl_sum = mmd_sum = correct = total = 0
        for batch in train_loader:
            X, y, d = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits, feat = model(X)
                loss_ce = criterion(logits, y)
                loss_cl = center_loss(feat.float(), y)
            # MMD 在 float32 下算（避免 half precision 问题）
            loss_mmd = mmd_loss(feat.float(), d, lambda_mmd=lambda_mmd)
            loss = loss_ce + loss_cl + loss_mmd

            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item(); ce_sum += loss_ce.item()
            cl_sum += loss_cl.item(); mmd_sum += loss_mmd.item()
            correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)
        n_batches = len(train_loader)
        if (epoch + 1) % 20 == 0 or val_acc > best_val:
            print(f"    Ep {epoch+1:03d} | CE {ce_sum/n_batches:.4f} CL {cl_sum/n_batches:.4f} MMD {mmd_sum/n_batches:.4f} | "
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


if __name__ == "__main__":
    all_files = [f'/mnt/eason_ckp/BCI/BCICIV_2a_gdf/A{i:02d}T_train.gdf' for i in range(1, 10)]
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    sample_data, _ = load_subject_data(all_files[0])
    n_times = sample_data.shape[2]; del sample_data; gc.collect()
    print(f"时间点数: {n_times}")

    prev = {
        'CL_only': [67.0, 42.7, 66.0, 46.2, 28.5, 41.7, 50.3, 68.8, 64.2],
        'V2_noCL': [66.7, 41.0, 61.5, 46.2, 35.1, 39.9, 41.3, 69.8, 65.3],
    }

    results = []

    for test_idx in range(9):
        print(f"\n{'='*60}")
        print(f"Fold {test_idx+1}/9 — 测试: A{test_idx+1:02d}")
        print(f"{'='*60}")

        # 加载源被试，记录 domain_id
        train_data_list, train_label_list, train_domain_list = [], [], []
        domain_counter = 0
        for i, f in enumerate(all_files):
            if i == test_idx: continue
            d, l = load_subject_data(f)
            d_aligned = euclidean_alignment(d)
            train_data_list.append(d_aligned)
            train_label_list.append(l)
            train_domain_list.append(np.full(len(l), domain_counter))
            domain_counter += 1
            del d, d_aligned

        train_data = np.concatenate(train_data_list)
        train_labels = np.concatenate(train_label_list)
        train_domains = np.concatenate(train_domain_list)
        del train_data_list, train_label_list, train_domain_list; gc.collect()

        test_data, test_labels = load_subject_data(all_files[test_idx])
        test_data = euclidean_alignment(test_data)

        mean = train_data.mean(axis=(0, 2), keepdims=True)
        std = train_data.std(axis=(0, 2), keepdims=True) + 1e-8
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std

        train_aug, labels_aug, domains_aug = augment_data_8x(train_data, train_labels, train_domains)
        del train_data, train_labels, train_domains; gc.collect()

        n = len(train_aug); perm = np.random.permutation(n); n_val = int(n * 0.15)
        X_train = torch.FloatTensor(train_aug[perm[n_val:]]).unsqueeze(1)
        y_train = torch.LongTensor(labels_aug[perm[n_val:]])
        d_train = torch.LongTensor(domains_aug[perm[n_val:]])
        X_val = torch.FloatTensor(train_aug[perm[:n_val]]).unsqueeze(1)
        y_val = torch.LongTensor(labels_aug[perm[:n_val]])
        X_test = torch.FloatTensor(test_data).unsqueeze(1)
        y_test = torch.LongTensor(test_labels)
        del train_aug, labels_aug, domains_aug, test_data, test_labels; gc.collect()

        print(f"  数据: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        train_loader = DataLoader(TensorDataset(X_train, y_train, d_train), batch_size=128,
                                  shuffle=True, pin_memory=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, pin_memory=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, pin_memory=True)

        model = EEGNet(n_channels=22, n_times=n_times).to(device)
        center_loss = CenterLoss(4, model.feat_dim, lambda_c=0.03).to(device)
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

        best_val, best_test = train_fold(
            model, center_loss, train_loader, val_loader, test_loader, device,
            num_epochs=300, lr=1e-3, lambda_mmd=0.1
        )

        cl = prev['CL_only'][test_idx]
        diff = best_test * 100 - cl
        print(f"  ✅ Fold {test_idx+1} | Test={best_test*100:.1f}% | CL_only={cl:.1f}% | Δ={diff:+.1f}%")
        results.append(best_test * 100)

        del model, center_loss, train_loader, val_loader, test_loader
        del X_train, y_train, d_train, X_val, y_val, X_test, y_test
        gc.collect(); torch.cuda.empty_cache()

    # ── 最终结果 ──
    print(f"\n{'='*60}")
    print("EEGNet + EA + Aug8x + CL + MMD + TemporalAttn — LOSO 结果:")
    print(f"{'='*60}")
    print(f"  {'Subj':<6} {'CL+MMD':>9} {'CL_only':>9} {'V2无CL':>9} {'Δ(vs CL)':>10}")
    print(f"  {'-'*45}")
    for i in range(9):
        subj = f"A{i+1:02d}"
        acc = results[i]; cl = prev['CL_only'][i]; v2 = prev['V2_noCL'][i]
        print(f"  {subj:<6} {acc:>8.1f}% {cl:>8.1f}% {v2:>8.1f}% {acc-cl:>+9.1f}%")

    mean_acc = np.mean(results); std_acc = np.std(results)
    cl_mean = np.mean(prev['CL_only']); v2_mean = np.mean(prev['V2_noCL'])
    print(f"\n  CL+MMD:  {mean_acc:.1f}% ± {std_acc:.1f}%")
    print(f"  CL_only: {cl_mean:.1f}%")
    print(f"  V2无CL:  {v2_mean:.1f}%")
    print(f"  vs CL:   {mean_acc - cl_mean:+.1f}%")
    print(f"{'='*60}")
