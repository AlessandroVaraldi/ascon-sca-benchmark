# ASCON 3-bit Subkey Recovery with Deep Learning

This repository contains a complete pipeline to perform **profiling side-channel attacks** on ASCON implementations using **3-bit subkeys** as labels.  
It trains and compares four models:

- A **baseline MLP** (defined in `run.py`)
- **CNN1D** (`models/cnn.py`)
- **TCN** (`models/tcn.py`)
- **TinyTransformer** (`models/transformer.py`)

The pipeline:

1. Uses the `random_keys` set for **profiling** (training/validation).
2. Uses the `fixed_keys` set for **attack** (key recovery).
3. Trains one classifier per 3-bit subkey index (`BitNum` from 0 to 63).
4. Uses a **multi-class SNR** to select an informative trace window per subkey.
5. Recovers all 3-bit subkeys for the fixed key.
6. Reconstructs the first 64 bits of the secret key using **all 3 bits** of each subkey and the known geometry of `select_subkey`.  
   The reconstruction combines the information from all 3-bit subkeys at the **bit level**, using their aggregated log-likelihoods.
7. Compares recovered key bits to the true fixed key and **ranks the models** using side-channel metrics (success rate, guessing entropy)
   together with key-bit similarity.

---

## Repository structure

```text
.
├── models/
│   ├── cnn.py           # CNN1D model (1D CNN for SCA traces)
│   ├── tcn.py           # TCN model (CNN + dilated residual blocks)
│   └── transformer.py   # TinyTransformer model (conv stem + self-attention)
│
├── loss_functions/
│   └── rankingloss.py   # Pairwise RankingLoss (RkL) as in Zaid et al.
│
├── run.py               # Main 3-bit subkey training & key recovery pipeline
└── run.sbatch           # Example SLURM sbatch script for HPC execution
```

## Requirements

* Python 3.8+
* Recommended environment: `conda` or `virtualenv`

Core Python dependencies (typical setup):

* `numpy`
* `h5py`
* `matplotlib`
* `torch` (PyTorch, with GPU support if available)
* `torchvision` (optional, but often present in PyTorch envs)

Install with `pip` (example):

```bash
pip install numpy h5py matplotlib torch
```

Or ensure your `conda` environment (e.g. `torch-gpu`) has these installed.

---

## Dataset format

The code expects an **HDF5 file** containing ASCON traces and metadata, with the following structure:

* Group `random_keys`

  * `traces`: shape `[N_random, T]` (float or int, converted to float32)
  * `metadata`: compound dtype with at least a field `key`
    (`metadata['key']` → `[N_random, key_len]` uint8)
* Group `fixed_keys`

  * `traces`: shape `[N_fixed, T]`
  * `metadata`: same format as above

Assumptions:

* **Unprotected** case:

  * `metadata['key']` already contains the full secret key bytes (no shares).
* All traces in `fixed_keys` use the **same secret key**.

The script uses `metadata['key']` to compute the **ground-truth 3-bit subkeys** via `ascon.select_subkey(bitnum, key)` and to derive the **true key bits** for evaluation.

---

## Models

The following models are supported:

* `mlp`: baseline fully-connected network defined in `run.py`.
  Input: flattened window `[B, W]` → logits `[B, 8]`.
* `cnn`: `CNN1D` in `models/cnn.py`.
  Input: `[B, 1, W]` → stack of Conv1d + BN + ReLU + MaxPool → global average pooling → classifier.
* `tcn`: `TCN` in `models/tcn.py`.
  Input: `[B, 1, W]` → CNN stem → TCN-style dilated residual blocks → global average pooling → classifier.
* `transformer`: `TinyTransformer` in `models/transformer.py`.
  Input: `[B, 1, W]` → Conv stem → positional mixing → stack of Transformer blocks (+ optional LoRA adapters) → attention pooling over time → classifier.

All models output logits of shape `[B, 8]` (8 classes for the 3-bit subkey).

---

## Loss functions

Two losses are available:

1. **Cross-entropy loss** (`--loss-type ce`)
   Standard multi-class classification loss.

2. **Ranking loss (RkL)** (`--loss-type ranking`)
   Implemented in `loss_functions/rankingloss.py`, based on:

   ```python
   L_i = sum_{k != y} log(1 + exp(-alpha * (s_y - s_k)))
   ```

   Here, `s_y` is the logit of the true class, and `s_k` the logit of a wrong class.

The loss type can be selected via the command-line argument `--loss-type`.

---

## How the pipeline works

For each model and for each subkey index `BitNum = 0..63`:

1. **Label computation**

   * Uses `metadata['key']` from `random_keys` and `fixed_keys`.
   * Calls `ascon.select_subkey(BitNum, key_bytes)` to obtain an integer in `[0..7]` (3 bits).
   * These values are used as **8-class labels** for training (`random_keys`) and for ground truth on the fixed key.

2. **Multi-class SNR and window selection**

   * Computes an SNR curve across time using the 8 subkey classes.
   * Selects a window of length `--window-size` centered on the maximum SNR.
   * This window is applied to both `random_keys` and `fixed_keys` traces.

3. **Training**

   * Uses `random_keys` with:

     * Inputs: windowed traces.
     * Labels: 3-bit subkey values `[0..7]`.
   * Splits profiling data into train/validation (80/20 by default).
   * Trains the model for `--epochs` epochs using the selected loss.
   * Saves:

     * `snr.png` (SNR curve + selected center)
     * `train_val_loss.png`
     * `train_val_acc.png`

4. **Attack on fixed_keys**

   * Applies the trained model to the windowed `fixed_keys` traces.
   * Aggregates log-softmax outputs over all traces to build log-likelihoods for each subkey hypothesis in `{0..7}`.
   * Selects the subkey with maximum total log-likelihood → **recovered 3-bit subkey for that BitNum**.

5. **Key reconstruction (first 64 bits)**

   * After all 64 subkeys are attacked, the script keeps the **aggregated log-likelihoods** over the 8 possible 3-bit values for each subkey.
   * Using the known structure of `ascon.select_subkey`, each 3-bit subkey is viewed as a constraint on three key bits.
   * For each key bit, the script combines the contributions of the three subkeys that contain it:
     it sums (in the log-domain) the likelihoods of all subkey classes compatible with the hypothesis `k_j = 0`
     and with `k_j = 1`, then decides using a log-likelihood ratio (bit-level probabilistic combination).
   * The resulting 64-bit vector is compared to the true key bits (derived from the first 8 key bytes).

6. **Evaluation and ranking**

   * For each model:

     * Computes **Guessing Entropy (GE)** as the average rank of the correct 3-bit subkey over the 64 BitNum,
     * Computes **Success Rate (SR)** as the fraction of subkeys where the correct 3-bit subkey is ranked first,
     * Computes key-bit similarity: `correct_bits / 64`,
     * Computes Hamming distance on the first 64 key bits: `64 − correct_bits`,
     * Saves a textual summary and plots.
   * At the end, a global ranking file (`global_ranking.txt`) is written, sorting models primarily by SR, then GE, then key-bit similarity.
---

## Running locally

Basic usage:

```bash
python run.py \
  --h5-path /path/to/ascon_unprotected.h5 \
  --output-dir ./results_ascon \
  --loss-type ce \
  --models mlp cnn tcn transformer \
  --epochs 50 \
  --batch-size 256 \
  --window-size 500
```

### Important arguments

* `--h5-path`
  Path to the HDF5 file containing `random_keys` and `fixed_keys`.

* `--output-dir`
  Base directory where all results (plots, summaries) will be written.

* `--loss-type`
  `ce` or `ranking`.

* `--models`
  Space-separated list of models to run (any subset of `mlp cnn tcn transformer`).

* `--epochs`
  Number of epochs per subkey and per model.

* `--batch-size`
  Batch size for training and evaluation.

* `--window-size`
  Length of the SNR-selected window (number of samples).  
  For each BitNum, the window is centered around the maximum of a **multi-class SNR** curve computed over the 8 subkey classes.

* `--seed`
  Random seed for reproducibility (default: 1234).

* `--device`
  Override the device (e.g. `cpu`, `cuda`, or `xpu`). If not set, the script automatically prefers `cuda` (if available),
  then `xpu` (if available), otherwise falls back to `cpu`.

---

## Running on an HPC (SLURM)

The file `run.sbatch` is an example SLURM submission script.
Typical usage:

```bash
sbatch run.sbatch
```

`run.sbatch`:

* Requests a GPU node (e.g. V100) with appropriate CPU, RAM, and time limits.
* Activates a `conda` environment (e.g. `torch-gpu`).
* Copies the project from `$SLURM_SUBMIT_DIR` to a per-job directory on `$SCRATCH`.
* Runs `run.py` with suitable arguments.
* Syncs all results back to a `results` directory in the original project folder.

Edit `run.sbatch` to adjust:

* `#SBATCH` directives (partition, time, memory, GPUs).
* `conda` environment name.
* `--h5-path`, `--epochs`, `--window-size`, `--loss-type`, `--models`.
* `RESULTS_DIR` or paths as needed for your cluster.

---

## Output structure

Under `--output-dir`, the script creates the following layout:

```text
<output-dir>/
  global_ranking.txt        # Final ranking of models by key-bit similarity

  model_MLP/
    summary.txt
    true_vs_recovered.png
    subkey_00/
      snr.png
      train_val_loss.png
      train_val_acc.png
    subkey_01/
      ...
    ...
  model_CNN/
    ...
  model_TCN/
    ...
  model_TRANSFORMER/
    ...
```

For each model:

* `summary.txt`
  Contains:

  * recovered subkeys (0..63),
  * per-subkey rank of the correct 3-bit subkey and per-model GE/SR,
  * true key bytes (first 8),
  * true vs recovered bit vectors on the first 64 bits,
  * Hamming distance and key-bit similarity.

* `true_vs_recovered.png`
  Step plot comparing true and recovered key bits (0..63).

* `subkey_xx/`
  Contains per-subkey diagnostic plots:

  * `snr.png`: multi-class SNR curve with the chosen center.
  * `train_val_loss.png`: training and validation loss per epoch.
  * `train_val_acc.png`: training and validation accuracy per epoch.

---

## Extending and customizing

Some natural extensions:

* Change model hyperparameters in `models/cnn.py`, `models/tcn.py`, or `models/transformer.py`.
* Adjust the baseline MLP architecture in `run.py`.
* Modify SNR-based window selection or use a fixed window.
* Integrate other loss functions in addition to cross-entropy and RankingLoss.
* Add protected-trace support by combining key/nonce shares before calling `select_subkey`.

All main configuration options relevant to experiments are exposed as command-line arguments to `run.py`.
