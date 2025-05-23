# GeCoSleep
GeCoSleep: A Generative Continual Learning Frame work for Cross-Center Adaptation in Sleep Staging

![structure _1_-1.png](https://s2.loli.net/2025/05/23/5OakgcuLVbRCjPi.png)

GeCoSleep employs a generative replay strategy to reconstruct the distribution of historical data without storing any raw samples, effectively mitigating catastrophic forgetting. 

These are the source code and experimental setup of GeCoSleep.

## Dataset
We conducted our experiments on four publicly available sleep datasets:

* ISRUC-S1: [https://sleeptight.isr.uc.pt/](https://sleeptight.isr.uc.pt/)
* SHHS: [https://sleepdata.org/datasets/shhs](https://sleepdata.org/datasets/shhs)
* MASS-SS3: [http://ceams-carsm.ca/en/MASS/](http://ceams-carsm.ca/en/MASS/)
* Sleep-EDF-153: [http://www.physionet.org/physiobank/database/sleep-edfx/](http://www.physionet.org/physiobank/database/sleep-edfx/)

These datasets were used to evaluate the performance of our proposed method. Please refer to the respective links for more information and access instructions.

## Requirements

## How to run
### 1. Parameter Description

Below are the available parameters in the script and their descriptions:S

## Dataset
We conducted our experiments on four publicly available sleep datasets:

* ISRUC-S1: [https://sleeptight.isr.uc.pt/](https://sleeptight.isr.uc.pt/)
* SHHS: [https://sleepdata.org/datasets/shhs](https://sleepdata.org/datasets/shhs)
* MASS-SS3: [http://ceams-carsm.ca/en/MASS/](http://ceams-carsm.ca/en/MASS/)
* Sleep-EDF-153: [http://www.physionet.org/physiobank/database/sleep-edfx/](http://www.physionet.org/physiobank/database/sleep-edfx/)

These datasets were used to evaluate the performance of our proposed method. Please refer to the respective links for more information and access instructions.

## Requirements

## How to run
### 1. Parameter Description

Below are the available parameters in the script and their descriptions:S

- `--path_prefix`: Dataset path prefix, default is `/home/ShareData`.
- `--random_seed`: Random seed, default is `42`.
- `--isruc1_path`: File path for the ISRUC-1 dataset, default is `ISRUC-1`.
- `--isruc1`: Channels for the ISRUC-1 dataset, default is `['C4_A1', 'LOC_A2']`.
- `--shhs_path`: File path for the SHHS dataset, default is `shhs1_process6`.
- `--shhs`: Channels for the SHHS dataset, default is `['EEG', 'EOG(L)']`.
- `--mass_path`: File path for the MASS dataset, default is `MASS_SS3_3000_25C-Cz`.
- `--mass`: Channels for the MASS dataset, default is `['C4', 'EogL']`.
- `--sleep_edf_path`: File path for the Sleep-EDF dataset, default is `sleep-edf-153-3chs`.
- `--sleep_edf`: Channels for the Sleep-EDF dataset, default is `['Fpz-Cz', 'EOG']`.
- `--normalize`: Whether to normalize samples, default is `True`.
- `--task_num`: Number of tasks, default is `4`.
- `--task_names`: List of task names, default is `['ISRUC1', 'SHHS', 'MASS', 'Sleep-EDF']`.
- `--cuda_idx`: CUDA device index, default is `0`.
- `--window_size`: Sequence length, default is `10`.
- `--total_num`: Number of examples for each task, default is `{'ISRUC1': 100, 'SHHS': 200, 'MASS': 60, 'Sleep-EDF': 150}`.
- `--fold_num`: Number of folds, default is `10`.
- `--num_epochs`: Total number of training epochs, default is `200`.
- `--batch_size`: Batch size, default is `32`.
- `--valid_epoch`: Validation interval, default is `5`.
- `--valid_batch`: Validation batch size, default is `32`.
- `--dropout`: Dropout rate, default is `0.15`.
- `--weight_decay`: Weight decay value, default is `1e-5`.
- `--lr`: Learning rate, default is `1e-4`.
- `--replay_mode`: Continual learning strategy, default is `none`.
- `--min_epoch`: Minimum number of epochs for model saving, default is `15`.
- `--num_epochs_generator`: Number of epochs for generator training, default is `100`.
- `--lr_seq_gen`: Learning rate for sequential generator, default is `1e-4`.
- `--beta`: Coefficient for KL loss, default is `0.1`.
- `--tau`: Temperature for knowledge distillation, default is `5`.
- `--gamma`: Updating rate for running loss, default is `1e-2`.
- `--enable_multihead`: Whether to enable multihead (LWF setting), default is `False`.
- `--ewc_lambda`: Coefficient for EWC penalty, default is `1e3`.
- `--ewc_gamma`: Updating rate for FIM, default is `0.4`.
- `--ewc_batches`: Number of batches for calculating FIM, default is `256`.
- `--replay_buffer`: Replay buffer size for each task, default is `128`.

### 2. Example Commands

Below are some example commands. You can modify the parameters as needed:

#### Basic Training Command

```bash
python main.py --random_seed 42 --cuda_idx 0 --num_epochs 200 --batch_size 32
```

#### Training with Generator

```bash
python main.py --replay_mode generative --num_epochs_generator 100 --lr_seq_gen 1e-4 --beta 0.1 --tau 5 --gamma 1e-2
```

#### Training with LWF Strategy

```bash
python main.py --replay_mode lwf --enable_multihead
```

#### Training with EWC Strategy

```bash
python main.py --replay_mode ewc --ewc_lambda 1e3 --ewc_gamma 0.4 --ewc_batches 256
```

#### Custom Dataset Paths and Channels

```bash
python main.py --isruc1 C4_A1 LOC_A2 --shhs EEG EOG(L)
```

### 3. Output

After each experiment, a directory will be generated in `/results`, containing:  
- Experiment result table (`.txt`)  
- Detailed experiment information (`.json`)  
- Model parameters (`.zip`)  

### 4. Notes

- Ensure all dataset paths are correct.
- All data should be preprocessed into `.mat` or `.npy`
- Adjust the `cuda_idx` and `batch_size` parameters according to your hardware configuration.
- Avoid running on windows.
