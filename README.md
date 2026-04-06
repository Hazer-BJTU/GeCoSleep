# GeCoSleep: Continual Learning for Sleep Staging Across Multiple Datasets

This repository contains the official implementation of the paper "GeCoSleep: A Generative Continual Learning Framework for Sleep Staging" (or similar). The project implements various continual learning strategies for sleep staging across multiple EEG datasets, including ISRUC1, SHHS, MASS, Sleep-EDF, PhysioNet, and HSP.

## Table of Contents
- [Installation](#installation)
- [Datasets Preparation](#datasets-preparation)
- [Quick Start](#quick-start)
- [Command Line Arguments](#command-line-arguments)
- [Continual Learning Methods and Hyperparameters](#continual-learning-methods-and-hyperparameters)
- [Output and Results](#output-and-results)
- [Project Structure](#project-structure)
- [Visualization](#visualization)
- [References](#references)

## Installation

### Prerequisites
- Python 3.8 or later
- PyTorch 1.9+ (with CUDA 11.3+ for GPU acceleration, optional but recommended)
- Conda (recommended) or pip

### Using Conda (Recommended)
Create a new conda environment and install the required packages:

```bash
conda create -n ge cosleep python=3.9
conda activate ge cosleep
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install numpy scipy pandas mne hmmlearn scikit-learn quadprog
```

Alternatively, you can install using the provided `environment.yml` file (contains many extra packages for visualization and development):

```bash
conda env create -f environment.yml
conda activate lcz0312
```

However, note that the `environment.yml` includes many system-level dependencies that may be specific to the original author's system. We recommend using the minimal installation above.

### Using Pip Only
If you prefer pip, ensure you have PyTorch with CUDA support installed from [pytorch.org](https://pytorch.org/get-started/locally/). Then run:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
pip install numpy scipy pandas mne hmmlearn scikit-learn quadprog
```
### Running on CPU
If you do not have a CUDA‑capable GPU, install CPU‑only PyTorch (e.g., `conda install pytorch cpuonly -c pytorch`). Then modify `clnetworks.py` line 32 to use `'cpu'` when `args.cuda_idx < 0`, or simply set `--cuda_idx 0` and ensure `torch.cuda.is_available()` returns `False` (PyTorch will fallback to CPU). However, the code currently does not handle CPU fallback automatically; you may need to edit the device assignment manually.

## Datasets Preparation

The code expects datasets to be placed under a common directory (default `/home/ShareData`). You can change the base path using the `--path_prefix` argument.

Each dataset must be preprocessed and stored in a specific format. Below are the expected structures and download links.

### ISRUC1
- **Expected path**: `{path_prefix}/ISRUC-1/ISRUC-1/`
- **Files**: `.mat` files containing EEG/EOG channels (C4_A1, LOC_A2, etc.) and corresponding label files in a `label` subdirectory.
- **Download**: [ISRUC-Sleep dataset](https://sleeptight.isr.uc.pt/ISRUC_Sleep/).

### SHHS
- **Expected path**: `{path_prefix}/shhs1_process6/`
- **Files**: `.pkl` files containing preprocessed data with keys `new_xall` and `stage_label`.
- **Download**: [Sleep Heart Health Study](https://sleepdata.org/datasets/shhs) (requires registration).

### MASS
- **Expected path**: `{path_prefix}/MASS_SS3_3000_25C-Cz/`
- **Files**: `.mat` files named `*-Datasub.mat` containing multi-channel EEG/EOG/EMG.
- **Download**: [MASS dataset](https://massdb.webs.upv.es/) (choose SS3 subset).

### Sleep-EDF
- **Expected path**: `{path_prefix}/sleep-edf-153-3chs/`
- **Files**: `.npz` files containing arrays `x` (shape [n_epochs, 3000, channels]) and `y` (labels). The channel order is `['Fpz-Cz', 'EOG', 'EMG']`.
- **Download**: [Sleep-EDF expanded](https://physionet.org/content/sleep-edfx/1.0.0/). You may need to preprocess to the required format.

### PhysioNet Challenge 2018
- **Expected path**: `{path_prefix}/PhysioNet-Challenge-2018_sub251_C4E1/`
- **Files**: Pairs of `.npy` files: `*_x.npy` (shape [n_epochs, channels, 3000]) and `*_y.npy` (labels). Channels are `['C4', 'E1']`.
- **Download**: [PhysioNet 2018](https://physionet.org/content/challenge-2018/1.0.0/). Preprocess to extract 30‑second epochs.

### HSP (Hospital Sleep Patterns)
- **Expected path**: `{path_prefix}/HSP_processed_0624_taiyang/`
- **Files**: `.fif` files (raw EEG) and `_annotations.csv` files (sleep stages). The code expects a subject directory `S0001` with subdirectories for each session.
- **Note**: This dataset may be private; contact the authors for access.

**Important**: The data loading functions assume each dataset has been preprocessed to have a fixed sampling rate of 100 Hz and epochs of 30 seconds (3000 samples). The code will resample if needed, but it's recommended to preprocess accordingly.
**Note on sample counts**: The `--total_num` argument is a dictionary that limits the number of subjects (or recordings) loaded per dataset. The default values are chosen for quick experimentation; you may increase them up to the total number of available subjects. For example, to use all ISRUC1 subjects (100), set `--total_num '{"ISRUC1":100,"SHHS":200,"MASS":60,"Sleep-EDF":150}'`. The exact maximum depends on your dataset size.

## Quick Start

After installing dependencies and placing datasets, you can run a basic experiment with the following command:

```bash
python main.py --path_prefix /path/to/your/datasets --cuda_idx 0 --replay_mode none --task_names ISRUC1 SHHS
```

This will run a 10‑fold cross‑validation continual learning experiment on the selected tasks using the vanilla fine‑tuning strategy (`replay_mode=none`). Results will be saved in the `results/` directory.

### Example: Run with Generative Replay
```bash
python main.py --path_prefix /home/ShareData --cuda_idx 0 --replay_mode generative --task_num 3 --task_names ISRUC1 SHHS MASS --batch_size 32 --num_epochs 200
```

### Example: Joint Training (all tasks merged)
```bash
python joint.py --path_prefix /home/ShareData --cuda_idx 0
```

## Command Line Arguments

The most important arguments are:

| Argument | Description | Default |
|----------|-------------|---------|
| `--path_prefix` | Base path to dataset directories | `/home/ShareData` |
| `--cuda_idx` | GPU device index | `0` |
| `--replay_mode` | Continual learning strategy (`none`, `generative`, `fine_tuning`, `independent`, `experience`, `lwf`, `ewc`, `der`, `dtw`, `tagem`, `agem`, `bayes`, `deep`) | `none` |
| `--task_names` | List of task names to include | `['ISRUC1','SHHS','MASS','Sleep-EDF']` |
| `--task_num` | Number of tasks (inferred from `task_names`) | `len(task_names)` |
| `--total_num` | Dictionary of sample counts per task | `{'ISRUC1':100,'SHHS':200,'MASS':60,'Sleep-EDF':150}` |
| `--window_size` | Sequence length (number of epochs per sample) | `10` |
| `--batch_size` | Training batch size | `32` |
| `--num_epochs` | Epochs per task | `200` |
| `--lr` | Learning rate | `1e-4` |
| `--dropout` | Dropout rate | `0.05` |
| `--weight_decay` | Weight decay (L2 penalty) | `1e-4` |
| `--random_seed` | Random seed for reproducibility | `42` |
| `--fold_num` | Number of cross‑validation folds | `10` |

For a full list of arguments (including strategy‑specific hyperparameters), see `main.py`.
**Note**: The `--total_num` dictionary must have keys exactly matching the task names (case‑sensitive). If you add or remove tasks, adjust the dictionary accordingly. The values represent the maximum number of subjects (recordings) to load from each dataset.
## Continual Learning Methods and Hyperparameters

The code implements several continual learning strategies. Below is a brief overview of each method and its specific hyperparameters (see `main.py` for defaults).

| Method (`--replay_mode`) | Description | Key Hyperparameters |
|--------------------------|-------------|---------------------|
| `none` | Vanilla fine‑tuning (no continual learning) | – |
| `fine_tuning` | Fine‑tuning with frozen feature extractor after first task | – |
| `independent` | Train separate model for each task | – |
| `experience` | Experience Replay (ER) with fixed‑size buffer | `--replay_buffer`: buffer size (default 512) |
| `generative` | **GeCoSleep** (Generative Continual Learning) with VAE + HMM | `--num_epochs_generator`: generator epochs (100)<br>`--lr_seq_gen`: generator LR (1e‑4)<br>`--beta`: VAE KL weight (0.1)<br>`--replay_lambda`: replay loss weight (10)<br>`--distill_lambda`: distillation weight (1)<br>`--tau`: distillation temperature (1)<br>`--distill_loss`: `'kl'`, `'ed'`, or `'mixed'`<br>`--mix_lambda`: mixing coefficient for `'mixed'` loss (0.5) |
| `lwf` | Learning without Forgetting (LwF) | `--tau`: distillation temperature (1) |
| `ewc` | Elastic Weight Consolidation (EWC) | `--ewc_lambda`: regularization strength (1000)<br>`--ewc_gamma`: FIM update rate (0.5)<br>`--ewc_batches`: # batches for FIM estimation (512) |
| `der` | Dark Experience Replay (DER) | `--der_alpha`: classification loss weight (0.5)<br>`--der_beta`: dark experience MSE weight (0.5) |
| `dtw` | Dynamic Time Warping (DT2W) | `--dtw_lambda`: DTW loss weight (0.03) |
| `tagem` | Task‑Aware Gradient Episodic Memory (TA‑GEM) | `--num_clusters`: # of memory clusters (8)<br>`--replay_buffer`: total buffer size (512) |
| `agem` | Averaged Gradient Episodic Memory (A‑GEM) | `--replay_buffer`: buffer size (512) |
| `bayes` | Bayesian EEGNet continual learning | Uses `--bayes_eeg_params` dictionary (see `main.py` lines 66‑70) |
| `deep` | DeepSleepNet‑based continual learning | – |

**Notes**:
- For `generative`, the method combines a sequential VAE (for feature replay) and an HMM (for label‑sequence generation). The hyperparameters control the trade‑off between reconstruction, KL divergence, and distillation losses.
- `experience`, `der`, `tagem`, and `agem` share the same replay buffer mechanism; the buffer is filled with past examples and used according to each algorithm's rules.
- `fine_tuning` freezes the CNN and short‑term encoder after the first task, only updating the long‑term encoder and classifier.
- `independent` trains a separate SleepNet for each task; no knowledge is transferred.
- `none` simply fine‑tunes the whole network on each new task (strong catastrophic forgetting expected).

All methods use the same base hyperparameters (`--lr`, `--batch_size`, `--num_epochs`, `--dropout`, etc.). The table above lists only the additional parameters that are specific to each strategy.

## Output and Results

### Directory Structure
After each run, a new directory is created under `results/` with a unique experiment ID (timestamp + random suffix). The directory will be named like `results/20250415_143022_abcd1234/`. Inside you will find:

### Output Files
1. **`cl_output_record_{replay_mode}.txt`** – Text summary of performance metrics:
   - Per‑task accuracy and macro‑F1 for each fold
   - Average accuracy and macro‑F1 across folds
   - Backward Transfer (BWT) and Forward Transfer (FWT) scores
   - Confusion matrices for each task

2. **`{experiment_id}.json`** – Detailed JSON log containing:
   - All command‑line arguments and hyperparameters
   - Per‑fold results (train/validation/test metrics at each epoch)
   - Confusion matrices for each task and fold
   - Timestamps and experiment metadata
   - Training loss curves and validation metrics history
   
   To inspect the JSON log, you can use:
   ```bash
   python -m json.tool results/20250415_143022_abcd1234/20250415_143022_abcd1234.json | less
   ```
   Or load it in Python:
   ```python
   import json
   with open('results/20250415_143022_abcd1234/20250415_143022_abcd1234.json', 'r') as f:
       log = json.load(f)
   ```

3. **`params.tar.gz`** – Archived trained model checkpoints (if saved). This tarball contains all `.pth` files saved during training. Each checkpoint is named as `{replay_mode}_task{task_num}_fold{fold_num}.pth`. To extract and inspect:
   ```bash
   tar -xzvf results/20250415_143022_abcd1234/params.tar.gz
   ```
   The extracted `.pth` files can be loaded in PyTorch:
   ```python
   import torch
   from models import SleepNet  # or appropriate model class
   model = SleepNet(num_channels=...)
   model.load_state_dict(torch.load('generative_task0_fold0.pth', map_location='cpu'))
   ```

### Real‑time Console Output
During training, the console prints:
- Per‑epoch training loss and validation accuracy
- Per‑fold progress (fold X/Y)
- Per‑task accuracy and macro‑F1 after each fold
- Model saving notifications

### Accessing Intermediate Model Weights
During training, model checkpoints are temporarily stored in `modelsaved/`. The best model for each task–fold combination is saved when validation accuracy improves. At the end of the experiment, all `.pth` files in `modelsaved/` are packed into `params.tar.gz` and the temporary files are deleted.

### Notes
- If `--save_model` is set to `False` (default is `True`), no `params.tar.gz` will be created.
- The JSON log is the most comprehensive record; it contains everything needed to reproduce plots and analyses.
- For quick evaluation of a saved model, see the `visualization/` notebooks for examples of loading and testing checkpoints.

## Project Structure

```
.
├── main.py                 # Entry point with argument parsing
├── train.py               # Main training loop and k‑fold evaluation
├── clnetworks.py          # Base continual learning network and strategies
├── models.py              # Sleep staging model (SleepNet) architecture
├── metric.py              # Confusion matrix and evaluation functions
├── data_preprocessing.py  # Data loading and preprocessing for all datasets
├── logs.py                # JSON logging utilities
├── joint.py               # Script for joint training (all tasks merged)
├── environment.yml        # Full conda environment spec (optional)
├── README.md              # This file
├── baselines/             # Implementations of baseline CL methods (EWC, LwF, DER, ...)
├── GeCoSleep/             # Generative continual learning components (VAE, HMM)
├── DeepSleepNet/          # DeepSleepNet model and CL wrapper
├── BayesEEGNet/           # Bayesian EEGNet model and CL wrapper
├── visualization/         # Jupyter notebooks for analysis (optional)
├── results/               # Output directory (created automatically)
└── modelsaved/            # Temporary model checkpoints (cleaned after archiving)
```

## Visualization

The `visualization/` directory contains Jupyter notebooks for analyzing and visualizing experimental results. These notebooks help understand feature distributions, model performance, and continual learning metrics.

### Notebooks Overview

| Notebook | Purpose | Dependencies |
|----------|---------|--------------|
| `select.ipynb` | Samples representative features using k‑means++ clustering. Loads `.pt` feature files from `datacache/` and saves sampled features to `sampled_features.npz`. | `scikit‑learn`, `torch`, `numpy`, `tqdm` |
| `decomposition.ipynb` | Reduces sampled features to 2D using PCA (50 components) + t‑SNE. Reads `sampled_features.npz` and writes `final_points.npz`. | `scikit‑learn`, `numpy` |
| `visualize.ipynb` | Plots 2D feature distributions for real features, generated features, and untrained‑generator features across tasks and sleep stages. Reads `final_points.npz`. | `matplotlib`, `numpy` |
| `statistical.ipynb` | Computes aggregated metrics (AP, BP, BWT, FWT, per‑task forgetting) from JSON result files in `results_json/`. Outputs `*_metrics.json` and LaTeX‑style tables. | `numpy`, `json`, `os` |
| `statistical2.ipynb` | Computes mean and standard deviation of accuracy/macro‑F1 across folds. Outputs `*_table.json`. | `numpy`, `json`, `os` |

### Visualization Pipeline

1. **Feature Caching** (prerequisite):  
   The notebooks expect a `datacache/` directory containing PyTorch `.pt` files of the form:
   - `task{t}_feature.pt` – real features extracted from the SleepNet encoder.
   - `task{t}_gen_feature.pt` – features generated by the trained VAE.
   - `task{t}_untrained_gen_feature.pt` – features from an untrained VAE.  
   These files are typically saved during generative‑replay experiments. If they are missing, you may need to modify the training code to dump features (e.g., add `torch.save` calls in `EEGGR.py`).

2. **Sampling** (`select.ipynb`):  
   Runs k‑means++ on each feature set to select 200 representative points per class. Saves `sampled_features.npz`.

3. **Dimensionality Reduction** (`decomposition.ipynb`):  
   Applies PCA followed by t‑SNE to obtain 2‑D coordinates for visualization. Saves `final_points.npz`.

4. **Plotting** (`visualize.ipynb`):  
   Creates a 3×4 grid of scatter plots (3 generator states × 4 tasks) showing sleep‑stage clusters.

### Result Analysis Pipeline

1. **Collect JSON logs**: Copy (or symlink) the JSON result files from `results/` into a `results_json/` directory at the project root.

2. **Run statistical notebooks**: Execute `statistical.ipynb` and `statistical2.ipynb` to generate summary JSON files and LaTeX‑ready tables.

### Usage Example

```bash
# Install additional visualization dependencies
pip install matplotlib scikit‑learn tqdm

# Create results_json directory and copy result files
mkdir -p results_json
cp results/*/*.json results_json/

# Run Jupyter notebooks (or execute cells in your preferred environment)
jupyter notebook visualization/statistical.ipynb
```

### Notes
- The `datacache/` directory is not included in the repository. You must generate feature files by running generative‑replay experiments with appropriate logging enabled.
- The notebooks assume 4 tasks and 5 sleep stages (W, N1, N2, N3, REM). If your experiment uses a different number of tasks, modify the `num_tasks` and `num_categories` variables accordingly.
- For a quick preview of the visualization output, you can check the existing `datacache/` files if available from prior runs.


## References

- **GeCoSleep paper** (to be added)
- **SleepNet model**: ...
- **Continual learning baselines**: EWC, LwF, DER, GEM, etc.

## Troubleshooting

- **CUDA out of memory**: Reduce `batch_size` or `window_size`.
- **Dataset not found**: Check `--path_prefix` and ensure the dataset folder matches the expected name.
- **Missing dependencies**: Install missing packages via pip. If `hmmlearn` fails on Windows, consider using WSL or conda.
- **Permission errors**: Ensure you have write permission for `results/` and `modelsaved/` directories.

For further questions, please open an issue on the repository.
