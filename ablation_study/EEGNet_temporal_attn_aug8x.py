"""
EEGNet + TemporalAttn 优化版 — 8x增强
基于 55.2% 的集成结果，将增强从4x恢复到8x：
1. Label Smoothing (0.1)
2. Mixup (alpha=0.2)
3. 8x增强（7种增强方式）
4. 5种子集成 Soft Voting

对比：4x增强集成(55.2%) vs 8x增强集成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mne
import random
import gc
import copy
from scipy.ndimage import gaussian_filter1d
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
    """8x增强，7种增强方式。预分配内存避免OOM"""
    n_trials, n_ch, n_times = data.shape
    n_aug = n_trials * 7  # 7种增强
    n_total = n_trials + n_aug

    # 预分配数组
    combined_data = np.empty((n_total, n_ch, n_times), dtype=data.dtype)
    combined_labels = np.empty(n_total, dtype=labels.dtype)
    combined_domains = np.empty(n_total, dtype=domain_ids.dtype) if domain_ids is not None else None

    # 原始数据
    combined_data[:n_trials] = data
    combined_labels[:n_trials] = labels
    if combined_domains is not None:
        combined_domains[:n_trials] = domain_ids

    idx = n_trials
    for i in range(n_trials):
        trial = data[i]; label = labels[i]; trial_std = np.std(trial)
        did = domain_ids[i] if domain_ids is not None else 0

        # 1. 高斯噪声
        combined_data[idx] = trial + 0.02 * trial_std * np.random.randn(n_ch, n_times)
        combined_labels[idx] = label
        if combined_domains is not None: combined_domains[idx] = did
        idx += 1

        # 2. 时间移位
        shift = random.choice([3, 5, 8]) * random.choice([-1, 1])
        combined_data[idx] = np.roll(trial, shift, axis=-1)
        combined_labels[idx] = label
        if combined_domains is not None: combined_domains[idx] = did
        idx += 1

        # 3. 通道dropout
        ch_drop = trial.copy()
        ch_drop[np.random.choice(n_ch, random.randint(2, 3), replace=False)] = 0
        combined_data[idx] = ch_drop
        combined_labels[idx] = label
        if combined_domains is not None: combined_domains[idx] = did
        idx += 1

        # 4. 幅度缩放
        combined_data[idx] = trial * np.random.uniform(0.8, 1.2)
        combined_labels[idx] = label
        if combined_domains is not None: combined_domains[idx] = did
        idx += 1

        # 5. 更强噪声
        combined_data[idx] = trial + 0.05 * trial_std * np.random.randn(n_ch, n_times)
        combined_labels[idx] = label
        if combined_domains is not None: combined_domains[idx] = did
        idx += 1

        # 6. 时间翻转
        combined_data[idx] = trial[:, ::-1]
        combined_labels[idx] = label
        if combined_domains is not None: combined_domains[idx] = did
        idx += 1

        # 7. 频带扰动（随机smooth）
        combined_data[idx] = gaussian_filter1d(trial, sigma=random.choice([2, 3, 5]), axis=-1)
        combined_labels[idx] = label
        if combined_domains is not None: combined_domains[idx] = did
        idx += 1

    perm = np.random.permutation(n_total)
    print(f"  增强: {n_trials} -> {n_total} ({n_total/n_trials:.1f}x)")
    if combined_domains is not None:
        return combined_data[perm], combined_labels[perm], combined_domains[perm]
    return combined_data[perm], combined_labels[perm], None


def mixup_data(x, y, alpha=0.2):
    """Mixup数据增强：在batch内随机配对混合"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam = max(lam, 1 - lam)  # 确保 lam >= 0.5
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class TemporalSelfAttention(nn.Module):
    def __init__(self, n_channels, time_steps, n_heads=2, dropout=0.15):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = n_channels // n_heads
        self.scale = self.head_dim ** 0.5
        self.qkv = nn.Linear(n_channels, 3 * n_channels, bias=False)
        self.out_proj = nn.Linear(n_channels, n_channels, bias=False)
        self.ln = nn.LayerNorm(n_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, _, t = x.shape
        x_seq = x.squeeze(2).permute(0, 2, 1)
        residual = x_seq
        qkv = self.qkv(x_seq)
        qkv = qkv.reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, t, c)
        out = self.out_proj(out)
        out = self.ln(out + residual)
        out = out.permute(0, 2, 1).unsqueeze(2)
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
        with torch.no_grad():
            dummy_conv = self._conv_features(torch.zeros(1, 1, n_channels, n_times))
            t_dim = dummy_conv.shape[3]
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
        x = self.temporal_attn(x)
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
    xx = torch.sum(x * x, dim=1, keepdim=True)
    yy = torch.sum(y * y, dim=1, keepdim=True)
    xy = torch.mm(x, y.t())
    dist = xx + yy.t() - 2 * xy
    return torch.exp(-dist / (2 * sigma ** 2))


def mmd_loss(feat, domain_ids, lambda_mmd=0.1):
    unique_domains = torch.unique(domain_ids)
    if len(unique_domains) < 2:
        return torch.tensor(0.0, device=feat.device)
    idx = torch.randperm(len(unique_domains))[:2]
    d1, d2 = unique_domains[idx[0]], unique_domains[idx[1]]
    feat1 = feat[domain_ids == d1]
    feat2 = feat[domain_ids == d2]
    if len(feat1) < 2 or len(feat2) < 2:
        return torch.tensor(0.0, device=feat.device)
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
    all_probs = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            with autocast('cuda'):
                logits, _ = model(X)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total, torch.cat(all_probs, dim=0)


def train_fold(model, center_loss, train_loader, val_loader, test_loader, device,
               num_epochs=300, lr=1e-3, lambda_mmd=0.1, use_mixup=True, mixup_alpha=0.2):
    # ✅ Label Smoothing = 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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

            # ✅ Mixup
            if use_mixup and random.random() < 0.5:  # 50%概率使用mixup
                X_mix, y_a, y_b, lam = mixup_data(X, y, alpha=mixup_alpha)
                with autocast('cuda'):
                    logits, feat = model(X_mix)
                    loss_ce = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                    loss_cl = lam * center_loss(feat.float(), y_a) + (1 - lam) * center_loss(feat.float(), y_b)
            else:
                with autocast('cuda'):
                    logits, feat = model(X)
                    loss_ce = criterion(logits, y)
                    loss_cl = center_loss(feat.float(), y)

            loss_mmd = mmd_loss(feat.float(), d, lambda_mmd=lambda_mmd)
            loss = loss_ce + loss_cl + loss_mmd

            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item(); ce_sum += loss_ce.item()
            cl_sum += loss_cl.item(); mmd_sum += loss_mmd.item()
            correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
        scheduler.step()
        val_acc, _ = evaluate(model, val_loader, device)
        test_acc, _ = evaluate(model, test_loader, device)
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

    # 旧结果对比
    prev_temporal_attn = [73.6, 38.2, 69.8, 51.0, 33.3, 39.6, 44.8, 73.3, 67.0]
    prev_cl_only = [67.0, 42.7, 66.0, 46.2, 28.5, 41.7, 50.3, 68.8, 64.2]
    prev_4x_ensemble = 55.2  # 4x增强集成的均值

    N_SEEDS = 5
    all_seed_results = []  # [seed][fold] = acc
    all_seed_probs = []    # [seed][fold] = probs tensor

    for seed_idx in range(N_SEEDS):
        seed = 42 + seed_idx * 100
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"\n{'#'*60}")
        print(f"# Seed {seed_idx+1}/{N_SEEDS} (seed={seed})")
        print(f"{'#'*60}")

        seed_results = []
        seed_probs = []

        for test_idx in range(9):
            print(f"\n{'='*60}")
            print(f"Seed {seed_idx+1} | Fold {test_idx+1}/9 — 测试: A{test_idx+1:02d}")
            print(f"{'='*60}")

            # 加载源被试
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

            # ✅ 8x增强
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
            if seed_idx == 0 and test_idx == 0:
                print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

            best_val, best_test = train_fold(
                model, center_loss, train_loader, val_loader, test_loader, device,
                num_epochs=300, lr=1e-3, lambda_mmd=0.1,
                use_mixup=True, mixup_alpha=0.2
            )

            # 获取概率用于集成
            _, test_probs = evaluate(model, test_loader, device)

            seed_results.append(best_test * 100)
            seed_probs.append(test_probs)

            print(f"  ✅ Seed{seed_idx+1} Fold{test_idx+1} | Test={best_test*100:.1f}% | 旧TempAttn={prev_temporal_attn[test_idx]:.1f}%")

            del model, center_loss, train_loader, val_loader, test_loader
            del X_train, y_train, d_train, X_val, y_val, X_test, y_test
            gc.collect(); torch.cuda.empty_cache()

        all_seed_results.append(seed_results)
        all_seed_probs.append(seed_probs)

        mean_acc = np.mean(seed_results)
        print(f"\n  Seed {seed_idx+1} 平均: {mean_acc:.1f}%")

    # ── 集成评估 ──
    print(f"\n{'='*70}")
    print("最终结果汇总")
    print(f"{'='*70}")

    # 每个seed的单模型结果
    print(f"\n各Seed单模型结果:")
    print(f"  {'Subj':<6}", end="")
    for s in range(N_SEEDS):
        print(f" {'Seed'+str(s+1):>8}", end="")
    print(f" {'集成':>8} {'旧TempAttn':>10}")
    print(f"  {'-'*(10+9*N_SEEDS+20)}")

    ensemble_results = []
    for fold in range(9):
        subj = f"A{fold+1:02d}"
        print(f"  {subj:<6}", end="")
        for s in range(N_SEEDS):
            print(f" {all_seed_results[s][fold]:>7.1f}%", end="")

        # Soft voting 集成
        # 加载测试标签
        test_data, test_labels = load_subject_data(all_files[fold])
        avg_probs = torch.zeros_like(all_seed_probs[0][fold])
        for s in range(N_SEEDS):
            avg_probs += all_seed_probs[s][fold]
        avg_probs /= N_SEEDS
        ensemble_preds = avg_probs.argmax(dim=1).numpy()
        ensemble_acc = (ensemble_preds == test_labels).mean() * 100
        ensemble_results.append(ensemble_acc)
        del test_data, test_labels

        print(f" {ensemble_acc:>7.1f}% {prev_temporal_attn[fold]:>9.1f}%")

    # 汇总
    print(f"\n  {'方案':<25} {'均值':>8} {'±std':>8}")
    print(f"  {'-'*45}")

    for s in range(N_SEEDS):
        m = np.mean(all_seed_results[s]); sd = np.std(all_seed_results[s])
        print(f"  {'Seed '+str(s+1)+' (单模型)':<25} {m:>7.1f}% {sd:>7.1f}%")

    # 所有单模型的平均
    all_single = [np.mean(all_seed_results[s]) for s in range(N_SEEDS)]
    print(f"  {'单模型平均':<25} {np.mean(all_single):>7.1f}% {np.std(all_single):>7.1f}%")

    m_ens = np.mean(ensemble_results); sd_ens = np.std(ensemble_results)
    print(f"  {'5种子集成 (Soft Vote)':<25} {m_ens:>7.1f}% {sd_ens:>7.1f}%")
    print(f"  {'旧TempAttn':<25} {np.mean(prev_temporal_attn):>7.1f}% {np.std(prev_temporal_attn):>7.1f}%")
    print(f"  {'旧CL_only':<25} {np.mean(prev_cl_only):>7.1f}% {np.std(prev_cl_only):>7.1f}%")

    print(f"\n  8x集成 vs 4x集成(55.2%): {m_ens - prev_4x_ensemble:+.1f}%")
    print(f"  8x集成 vs 旧TempAttn:    {m_ens - np.mean(prev_temporal_attn):+.1f}%")
    print(f"  8x集成 vs 旧CL_only:     {m_ens - np.mean(prev_cl_only):+.1f}%")
    print(f"{'='*70}")
