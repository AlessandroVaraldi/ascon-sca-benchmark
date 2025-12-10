import argparse
import os
import math
from typing import Dict, List, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Import ASCON helpers (for select_subkey)
import ascon_helper as ascon

# Import models
from models.cnn import CNN1D
from models.tcn import TCN
from models.transformer import TinyTransformer

# Import RankingLoss
from loss_functions.rankingloss import RankingLoss


# ---------------------------------------------------------------------------
# Utility: reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 1234) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Baseline MLP model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Simple baseline MLP for SCA traces (flattened window).

    Input:  [B, T]  (float)
    Output: [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 8,
        hidden_dims: List[int] = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] or [B, 1, T]
        if x.dim() == 3 and x.shape[1] == 1:
            x = x[:, 0, :]  # [B, T]
        return self.net(x)


# ---------------------------------------------------------------------------
# Dataset and SNR-based windowing
# ---------------------------------------------------------------------------

class SCASubkeyDataset(Dataset):
    """
    PyTorch Dataset for a single 3-bit subkey classification task.

    - traces: numpy [N, T]
    - labels: numpy [N] (values in [0..7])
    - mean/std: float or numpy arrays for normalization
    """

    def __init__(
        self,
        traces: np.ndarray,
        labels: np.ndarray,
        mean: np.ndarray = None,
        std: np.ndarray = None,
    ):
        assert traces.shape[0] == labels.shape[0]
        self.traces = traces.astype(np.float32)
        self.labels = labels.astype(np.int64)

        if mean is None or std is None:
            # global mean/std over dataset
            self.mean = self.traces.mean(axis=0)
            self.std = self.traces.std(axis=0) + 1e-9
        else:
            self.mean = mean
            self.std = std

        # Normalize in-place
        self.traces = (self.traces - self.mean) / self.std

    def __len__(self) -> int:
        return self.traces.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.traces[idx]  # [T]
        y = self.labels[idx]  # scalar
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def compute_multi_class_snr(
    traces: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 8,
) -> np.ndarray:
    """
    Compute a generalized SNR curve for multi-class labels.

    For each time sample t:
        SNR(t) = Var_c( mu_c(t) ) / Mean_c( sigma_c^2(t) )

    where:
        - mu_c(t) = mean of traces at time t for class c
        - sigma_c^2(t) = variance of traces at time t for class c

    Args:
        traces: [N, T]
        labels: [N] in [0..num_classes-1]
    Returns:
        snr: [T] SNR curve
    """
    N, T = traces.shape
    mu_c = []
    var_c = []

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            # No samples for this class; skip
            continue
        class_traces = traces[idx]  # [Nc, T]
        mu_c.append(class_traces.mean(axis=0))
        var_c.append(class_traces.var(axis=0) + 1e-12)

    mu_c = np.stack(mu_c, axis=0)      # [C_eff, T]
    var_c = np.stack(var_c, axis=0)    # [C_eff, T]

    mu_global = mu_c.mean(axis=0)      # [T]
    snr_num = ((mu_c - mu_global) ** 2).mean(axis=0)  # Var of means
    snr_den = var_c.mean(axis=0)                      # Mean of variances

    snr = snr_num / (snr_den + 1e-12)
    return snr


def select_snr_window(
    traces: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    num_classes: int = 8,
    out_path: str = None,
) -> Tuple[int, int]:
    """
    Compute SNR for the given traces/labels and select a window of length
    'window_size' centered around the maximum SNR point.

    Optionally saves a plot of the SNR curve at out_path.
    """
    N, T = traces.shape
    window_size = min(window_size, T)

    snr = compute_multi_class_snr(traces, labels, num_classes=num_classes)  # [T]
    center = int(np.argmax(snr))
    half = window_size // 2

    start = max(0, center - half)
    end = min(T, start + window_size)
    start = max(0, end - window_size)  # ensure exact length if possible

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.figure()
        plt.plot(snr)
        plt.axvline(center, linestyle="--")
        plt.title("SNR curve (multi-class)")
        plt.xlabel("Sample index")
        plt.ylabel("SNR")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    return start, end


# ---------------------------------------------------------------------------
# Subkey extraction from metadata[key]
# ---------------------------------------------------------------------------

def extract_combined_keys_from_metadata(metadata: np.ndarray) -> np.ndarray:
    """
    Extract 'key' field from metadata (unprotected case).

    metadata: HDF5 dataset with a compound dtype
    returns: numpy array [N, key_bytes]
    """
    # metadata['key'] is a (N, key_len) uint8 array
    keys = np.array(metadata['key'], dtype=np.uint8)
    return keys  # [N, key_len]


def compute_subkeys_for_all_bitnums(
    combined_keys: np.ndarray,
    num_bitnums: int = 64,
) -> np.ndarray:
    """
    For each trace and each BitNum in [0..num_bitnums-1], compute the 3-bit
    subkey label using ascon.select_subkey.

    combined_keys: [N, key_len] uint8
    returns: subkeys [N, num_bitnums] with values in [0..7]
    """
    N, key_len = combined_keys.shape
    subkeys = np.zeros((N, num_bitnums), dtype=np.int64)

    for i in range(N):
        key_bytes = combined_keys[i]  # uint8 array
        for bitnum in range(num_bitnums):
            subkeys[i, bitnum] = ascon.select_subkey(bitnum, key_bytes, array=False)

    return subkeys


# ---------------------------------------------------------------------------
# Key reconstruction (first 64 bits) from recovered subkeys
# ---------------------------------------------------------------------------

def reconstruct_key_bits_from_subkeys(
    recovered_subkeys: np.ndarray,
    bit_length: int = 64,
) -> np.ndarray:
    """
    Reconstruct a 64-bit key from recovered 3-bit subkeys.

    For i < 64, ascon.select_subkey(i, key) builds:
        sub_key[0] = key bit at position i
        sub_key[1] = key bit at position (i + dr[reg_num_0][0]) % 64
        sub_key[2] = key bit at position (i + dr[reg_num_0][1]) % 64

    Therefore, each key bit j in [0..63] appears exactly 3 times:
        - as sub_key[0] of subkey j
        - as sub_key[1] of subkey (j - dr[reg_num_0][0]) % 64
        - as sub_key[2] of subkey (j - dr[reg_num_0][1]) % 64

    We decode the 3 bits for every recovered subkey and, for each key bit j,
    perform a majority vote over the three occurrences to get the final bit.

    Args:
        recovered_subkeys: [num_bitnums], each in [0..7]
        bit_length: number of key bits to reconstruct (<= num_bitnums)

    Returns:
        key_bits: [bit_length] array with bits in {0,1}
    """
    num_bitnums = recovered_subkeys.shape[0]
    assert bit_length <= num_bitnums, "bit_length cannot exceed num_bitnums"

    # Decode the 3 bits of each subkey: s = b0<<2 | b1<<1 | b2
    b0 = np.zeros(num_bitnums, dtype=np.int64)
    b1 = np.zeros(num_bitnums, dtype=np.int64)
    b2 = np.zeros(num_bitnums, dtype=np.int64)
    for i, s_val in enumerate(recovered_subkeys):
        s = int(s_val)
        b0[i] = (s >> 2) & 1
        b1[i] = (s >> 1) & 1
        b2[i] = s & 1

    # Get the shifts used in select_subkey from ascon_helper
    shift1, shift2 = ascon.dr[ascon.reg_num_0]   # e.g. (57, 23)

    key_bits = np.zeros(bit_length, dtype=np.int64)
    for j in range(bit_length):
        # Occurrences of key bit j:
        #   - b0[j]
        #   - b1[(j - shift1) % 64]
        #   - b2[(j - shift2) % 64]
        v0 = b0[j]
        v1 = b1[(j - shift1) % bit_length]
        v2 = b2[(j - shift2) % bit_length]

        votes_sum = v0 + v1 + v2
        key_bits[j] = 1 if votes_sum >= 2 else 0

    return key_bits


def key_bytes_to_bits_first64(key_bytes: np.ndarray) -> np.ndarray:
    """
    Convert first 8 bytes (64 bits) of key into a bit vector [64],
    big-endian bit order (bit 0 = MSB of first byte).
    """
    assert key_bytes.ndim == 1
    assert key_bytes.size >= 8
    first8 = bytes(key_bytes[:8])
    val = int.from_bytes(first8, byteorder="big")
    bits = np.zeros(64, dtype=np.int64)
    for i in range(64):
        bits[i] = (val >> (63 - i)) & 1
    return bits


# ---------------------------------------------------------------------------
# Training loop for one subkey and one model
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    input_length: int,
    num_classes: int = 8,
) -> nn.Module:
    """
    Construct a model given its name.
    """
    if model_name.lower() == "mlp":
        return MLP(input_dim=input_length, num_classes=num_classes)

    elif model_name.lower() == "cnn":
        return CNN1D(
            in_channels=1,
            num_classes=num_classes,
            base_channels=32,
            num_conv_layers=4,
            kernel_size=11,
            pool_size=2,
            dropout=0.5,
            use_input_avgpool=False,
            input_pool_kernel=2,
        )

    elif model_name.lower() == "tcn":
        return TCN(
            in_channels=1,
            num_classes=num_classes,
            base_channels=32,
            num_conv_layers=3,
            kernel_size=11,
            pool_size=2,
            dropout=0.5,
            use_input_avgpool=True,
            input_pool_kernel=2,
            tcn_depth=3,
            tcn_kernel_size=5,
            tcn_dropout=0.1,
        )

    elif model_name.lower() == "transformer":
        return TinyTransformer(
            in_ch=1,
            d_model=96,
            nhead=4,
            depth=3,
            ffn_mult=2,
            num_classes=num_classes,
            dropout=0.1,
            lora_r=0,
            lora_alpha=8,
            lora_dropout=0.0,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def train_one_subkey(
    model_name: str,
    loss_type: str,
    traces_train: np.ndarray,
    labels_train: np.ndarray,
    traces_attack: np.ndarray,
    labels_attack_subkeys: np.ndarray,  # shape [N_attack] (true subkey, 0..7)
    output_dir: str,
    bitnum: int,
    batch_size: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
) -> Tuple[int, int, float, List[float], List[float], List[float], List[float]]:
    """
    Train a single model on a single BitNum (3-bit subkey classification).

    Returns:
        best_recovered_subkey (int),
        rank_true (int, in [1..8]),         # rank of the correct subkey in attack log-likelihoods
        sr (float, 0.0 or 1.0),             # success indicator (1 if rank_true == 1)
        train_losses, val_losses, train_accs, val_accs
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Build Dataset (train/val split) with normalization from train.
    dataset_full = SCASubkeyDataset(traces_train, labels_train)
    N = len(dataset_full)
    val_size = max(1, int(0.2 * N))
    train_size = N - val_size
    dataset_train, dataset_val = random_split(
        dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Use the same normalization for attack data
    mean = dataset_full.mean
    std = dataset_full.std
    dataset_attack = SCASubkeyDataset(traces_attack, labels_attack_subkeys, mean=mean, std=std)

    # 2) Dataloaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)
    attack_loader = DataLoader(dataset_attack, batch_size=batch_size, shuffle=False, drop_last=False)

    # 3) Build model
    input_length = traces_train.shape[1]
    num_classes = 8
    model = build_model(model_name, input_length=input_length, num_classes=num_classes)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "ranking":
        criterion = RankingLoss(alpha=5.0)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # 4) Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
        # ---- Train (accuracy here is diagnostic only, attack metrics are GE/SR) ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)          # [B,T]
            y = y.to(device)          # [B]

            if model_name.lower() in ["cnn", "tcn", "transformer"]:
                x_in = x.unsqueeze(1) # [B,1,T]
            else:
                x_in = x              # [B,T]

            logits = model(x_in)      # [B,C]
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---- Validation ----
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                if model_name.lower() in ["cnn", "tcn", "transformer"]:
                    x_in = x.unsqueeze(1)
                else:
                    x_in = x

                logits = model(x_in)
                loss = criterion(logits, y)

                running_loss += loss.item() * x.size(0)
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        val_loss = running_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"[{model_name}][BitNum {bitnum:02d}] "
            f"Epoch {epoch}/{epochs} - "
            f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

    # 5) Plot training curves (for debugging / diagnostics)
    epochs_range = range(1, epochs + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train")
    plt.plot(epochs_range, val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - BitNum {bitnum} - Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_val_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs_range, train_accs, label="Train")
    plt.plot(epochs_range, val_accs, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - BitNum {bitnum} - Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_val_acc.png"))
    plt.close()

    # 6) Attack: aggregate log-probabilities over fixed_keys traces to estimate subkey
    model.eval()
    num_classes = 8
    log_likelihoods = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for x, _ in attack_loader:
            x = x.to(device)

            if model_name.lower() in ["cnn", "tcn", "transformer"]:
                x_in = x.unsqueeze(1)
            else:
                x_in = x

            logits = model(x_in)  # [B, C]
            log_probs = F.log_softmax(logits, dim=1)  # [B, C]
            log_likelihoods += log_probs.sum(dim=0).cpu().numpy()

    # Most likely subkey (rank 1)
    best_recovered_subkey = int(np.argmax(log_likelihoods))

    # Guessing Entropy (rank of true subkey) and Success Rate indicator
    # All fixed_keys traces share the same true subkey for this BitNum
    true_subkey = int(labels_attack_subkeys[0])
    # ranks: indices sorted by log-likelihood (descending)
    ranks = np.argsort(log_likelihoods)[::-1]
    # position of the true subkey in ranks (0-based) -> GE is 1-based
    rank_true = int(np.where(ranks == true_subkey)[0][0]) + 1
    sr = 1.0 if rank_true == 1 else 0.0

    return best_recovered_subkey, rank_true, sr, train_losses, val_losses, train_accs, val_accs



# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3-bit subkey profiling & key recovery on ASCON traces.")
    parser.add_argument("--h5-path", type=str, required=True, help="Path to HDF5 traces file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Base output directory.")
    parser.add_argument("--loss-type", type=str, choices=["ce", "ranking"], default="ce",
                        help="Loss function: 'ce' for CrossEntropy, 'ranking' for RankingLoss.")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["mlp", "cnn", "tcn", "transformer"],
                        help="List of models to run: mlp cnn tcn transformer.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs per subkey.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--window-size", type=int, default=500, help="Window size (number of samples).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. 'cpu', 'cuda', or 'xpu'). "
             "If not set, prefer CUDA, then XPU, then CPU.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    # ----------------------------------------------------------------------
    # Load HDF5
    # ----------------------------------------------------------------------
    h5 = h5py.File(args.h5_path, "r")

    grp_random = h5["random_keys"]
    grp_fixed = h5["fixed_keys"]

    traces_random = np.array(grp_random["traces"])  # [N_rdm, T]
    traces_fixed = np.array(grp_fixed["traces"])    # [N_fixed, T]

    metadata_random = np.array(grp_random["metadata"])
    metadata_fixed = np.array(grp_fixed["metadata"])

    print(f"Random keys traces: {traces_random.shape}")
    print(f"Fixed  keys traces: {traces_fixed.shape}")

    # ----------------------------------------------------------------------
    # Extract keys (unprotected case) and compute true subkeys for all BitNum
    # ----------------------------------------------------------------------
    combined_keys_random = extract_combined_keys_from_metadata(metadata_random)  # [N_rdm, key_len]
    combined_keys_fixed = extract_combined_keys_from_metadata(metadata_fixed)    # [N_fixed, key_len]

    num_bitnums = 64
    print("Computing true 3-bit subkeys for random_keys...")
    subkeys_random = compute_subkeys_for_all_bitnums(combined_keys_random, num_bitnums=num_bitnums)
    print("Computing true 3-bit subkeys for fixed_keys...")
    subkeys_fixed = compute_subkeys_for_all_bitnums(combined_keys_fixed, num_bitnums=num_bitnums)

    # True key (we assume all fixed_keys share the same key)
    true_key_bytes_fixed = combined_keys_fixed[0]  # [key_len]
    true_key_bits64 = key_bytes_to_bits_first64(true_key_bytes_fixed)  # [64]

    # ----------------------------------------------------------------------
    # For each model: train/attack on each BitNum, reconstruct key, compare
    # ----------------------------------------------------------------------
    model_results: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name in args.models:
        model_name_clean = model_name.lower()
        model_dir = os.path.join(args.output_dir, f"model_{model_name_clean.upper()}")
        os.makedirs(model_dir, exist_ok=True)

        recovered_subkeys_all = np.zeros(num_bitnums, dtype=np.int64)
        ranks_all = np.zeros(num_bitnums, dtype=np.int64)
        sr_all = np.zeros(num_bitnums, dtype=np.float64)

        print(f"\n========== Model: {model_name_clean.upper()} ==========")

        for bitnum in range(num_bitnums):
            print(f"\n--- BitNum {bitnum:02d} ---")

            # Labels for this subkey
            y_random = subkeys_random[:, bitnum]  # [N_rdm]
            y_fixed = subkeys_fixed[:, bitnum]    # [N_fixed]

            # SNR-based window selection on training set
            subkey_dir = os.path.join(model_dir, f"subkey_{bitnum:02d}")
            snr_plot_path = os.path.join(subkey_dir, "snr.png")

            start, end = select_snr_window(
                traces_random,
                y_random,
                window_size=args.window_size,
                num_classes=8,
                out_path=snr_plot_path,
            )
            print(f"Selected window: [{start}, {end}) for BitNum {bitnum}")

            X_random_win = traces_random[:, start:end]  # [N_rdm, W]
            X_fixed_win = traces_fixed[:, start:end]    # [N_fixed, W]

            # Train and attack
            best_subkey, rank_true, sr, train_losses, val_losses, train_accs, val_accs = train_one_subkey(
                model_name=model_name_clean,
                loss_type=args.loss_type,
                traces_train=X_random_win,
                labels_train=y_random,
                traces_attack=X_fixed_win,
                labels_attack_subkeys=y_fixed,
                output_dir=subkey_dir,
                bitnum=bitnum,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=1e-3,
                device=device,
            )

            recovered_subkeys_all[bitnum] = best_subkey
            ranks_all[bitnum] = rank_true
            sr_all[bitnum] = sr
            print(
                f"[{model_name_clean.upper()}][BitNum {bitnum:02d}] "
                f"Recovered subkey: {best_subkey} | rank_true: {rank_true} | SR: {sr:.1f}"
            )

        # ------------------------------------------------------------------
        # Reconstruct key bits from recovered subkeys and compare
        # ------------------------------------------------------------------
        recovered_key_bits64 = reconstruct_key_bits_from_subkeys(recovered_subkeys_all, bit_length=64)

        # Similarity metrics  on key bits
        correct_bits = (recovered_key_bits64 == true_key_bits64).sum()
        total_bits = 64
        similarity = correct_bits / total_bits
        hamming_distance = total_bits - correct_bits

        # Guessing Entropy (GE) and Success Rate (SR) across all subkeys
        ge_model = float(ranks_all.mean())   # average rank of the true subkey (1..8)
        sr_model = float(sr_all.mean())      # fraction of subkeys with rank 1

        print(
            f"\n[{model_name_clean.upper()}] "
            f"GE (avg rank over subkeys): {ge_model:.3f} | "
            f"SR (subkey success rate): {sr_model:.3f} | "
            f"Key-bit similarity: {similarity:.4f} ({correct_bits}/{total_bits} bits correct, "
            f"Hamming distance: {hamming_distance})"
        )

        # Save summary
        summary_path = os.path.join(model_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Model: {model_name_clean.upper()}\n")
            f.write(f"Loss type: {args.loss_type}\n")
            f.write(f"Guessing Entropy (avg rank over 64 subkeys): {ge_model:.6f}\n")
            f.write(f"Success Rate (fraction of subkeys with rank 1): {sr_model:.6f}\n\n")
            f.write(f"Recovered subkeys (0..63):\n{recovered_subkeys_all.tolist()}\n\n")
            f.write(f"Per-subkey ranks (1..8, true subkey rank):\n{ranks_all.tolist()}\n\n")
            f.write(f"True key (first 8 bytes): {true_key_bytes_fixed[:8].tolist()}\n")
            f.write(f"True key bits[0..63]: {true_key_bits64.tolist()}\n")
            f.write(f"Recovered key bits[0..63]: {recovered_key_bits64.tolist()}\n")
            f.write(f"Correct bits: {correct_bits}/{total_bits}\n")
            f.write(f"Hamming distance: {hamming_distance}\n")
            f.write(f"Similarity: {similarity:.6f}\n")

        # Plot true vs recovered bits
        plt.figure(figsize=(10, 3))
        plt.step(range(total_bits), true_key_bits64, where="mid", label="True", linewidth=2)
        plt.step(range(total_bits), recovered_key_bits64, where="mid", label="Recovered", linestyle="--")
        plt.ylim(-0.2, 1.2)
        plt.xlabel("Bit index (0..63)")
        plt.ylabel("Bit value")
        plt.title(f"{model_name_clean.upper()} - True vs Recovered key bits")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "true_vs_recovered.png"))
        plt.close()

        # Store results for final ranking
        model_results[model_name_clean] = {
            "recovered_subkeys": recovered_subkeys_all,
            "recovered_key_bits": recovered_key_bits64,
            "similarity": np.array([similarity]),
            "hamming_distance": np.array([hamming_distance]),
            "ge": np.array([ge_model]),
            "sr": np.array([sr_model]),
        }

    h5.close()

    # ----------------------------------------------------------------------
    # Final ranking of models by SR, GE, then key-bit similarity
    # ----------------------------------------------------------------------
    print("\n================ Final Model Ranking (by SR, GE, then key-bit similarity) ================")
    ranking = sorted(
        model_results.items(),
        key=lambda kv: (
            -float(kv[1]["sr"][0]),          # higher SR is better
            +float(kv[1]["ge"][0]),          # lower GE is better
            -float(kv[1]["similarity"][0]),  # higher similarity is better
        ),
    )

    for rank, (model_name, res) in enumerate(ranking, start=1):
        sim = float(res["similarity"][0])
        hd = int(res["hamming_distance"][0])
        ge = float(res["ge"][0])
        sr = float(res["sr"][0])
        print(f"{rank}. {model_name.upper():>12} - SR {sr:.4f}, GE {ge:.4f}, similarity {sim:.4f}, Hamming distance {hd}")

    # Save a global summary file with SR, GE and key-bit similarity
    global_summary_path = os.path.join(args.output_dir, "global_ranking.txt")
    with open(global_summary_path, "w") as f:
        f.write("Model ranking (SR, GE, key-bit similarity on first 64 bits of key):\n\n")
        for rank, (model_name, res) in enumerate(ranking, start=1):
            sim = float(res["similarity"][0])
            hd = int(res["hamming_distance"][0])
            ge = float(res["ge"][0])
            sr = float(res["sr"][0])
            f.write(f"{rank}. {model_name.upper():>12} - SR {sr:.6f}, GE {ge:.6f}, similarity {sim:.6f}, Hamming distance {hd}\n")


if __name__ == "__main__":
    main()
