import numpy as np
import pickle as pkl
import scipy.io as sio
import pandas as pd
import random
import torch
import mne
import os
from scipy import signal
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def load_data_isruc1(filepath, window_size, channels, total_num, normalize):
    file_names = [file for file in os.listdir(filepath) if file.endswith('.mat')]
    file_names.sort()
    datas, labels = [], []
    for file in file_names:
        print(f'loading raw data from {os.path.join(filepath, file)}...')
        raw_data = sio.loadmat(os.path.join(filepath, file))
        X = None
        for channel in channels:
            data_resampled = signal.resample(raw_data[channel], 3000, axis=1)
            if normalize:
                mu, sigma = np.mean(data_resampled), np.std(data_resampled)
                data_resampled = (data_resampled - mu) / sigma
            '''
            print(f'calculating stft for channel {channel} in isruc1...')
            _, _, Zxx = signal.stft(data_resampled, 100, 'hann', 256)
            Zxx = 20 * np.log10(np.abs(Zxx))
            '''
            if X is None:
                X = torch.unsqueeze(torch.tensor(data_resampled, dtype=torch.float32, requires_grad=False), dim=1)
            else:
                temp = torch.unsqueeze(torch.tensor(data_resampled, dtype=torch.float32, requires_grad=False), dim=1)
                X = torch.cat((X, temp), dim=1)
        label_name = file.split('.')[0][7:] + '_1.npy'
        label = np.load(os.path.join(filepath, 'label', label_name))
        y = torch.tensor(label, dtype=torch.int64, requires_grad=False)
        data_seq, label_seq, segs = [], [], X.shape[0] // window_size
        for idx in range(segs):
            data_seq.append(X[idx * window_size: (idx + 1) * window_size])
            label_seq.append(y[idx * window_size: (idx + 1) * window_size])
        datas.append(data_seq)
        labels.append(label_seq)
        if len(datas) >= total_num:
            print('sufficient data loaded...')
            break
    return datas, labels


def load_data_shhs(filepath, window_size, channels, total_num, normalize):
    file_names = [file for file in os.listdir(filepath) if file.endswith('.pkl')]
    file_names.sort()
    shhs_channels = ['EEG', "EEG(sec)", 'EOG(L)', 'EMG']
    channel_index = [shhs_channels.index(c) for c in channels]
    datas, labels = [], []
    for file in file_names:
        print(f'loading raw data from {os.path.join(filepath, file)}...')
        with open(os.path.join(filepath, file), 'rb') as data_file:
            raw_data = pkl.load(data_file)
        raw_data_trans = raw_data['new_xall'][:, channel_index]
        sleep_epoch_num = raw_data_trans.shape[0] // 3000
        raw_data_trans = raw_data_trans.transpose(1, 0)
        X = None
        for idx in range(raw_data_trans.shape[0]):
            series = raw_data_trans[idx]
            series = series.reshape(sleep_epoch_num, 3000)
            if normalize:
                mu, sigma = np.mean(series), np.std(series)
                series = (series - mu) / sigma
            '''
            print(f'calculating stft for channel index {idx} in shhs...')
            _, _, Zxx = signal.stft(series, 100, 'hann', 256)
            Zxx = 20 * np.log10(np.abs(Zxx))
            '''
            if X is None:
                X = torch.unsqueeze(torch.tensor(series, dtype=torch.float32, requires_grad=False), dim=1)
            else:
                temp = torch.unsqueeze(torch.tensor(series, dtype=torch.float32, requires_grad=False), dim=1)
                X = torch.cat((X, temp), dim=1)
        y = torch.tensor(raw_data['stage_label'], dtype=torch.int64, requires_grad=False)
        data_seq, label_seq, segs = [], [], X.shape[0] // window_size
        for idx in range(segs):
            data_seq.append(X[idx * window_size: (idx + 1) * window_size])
            label_seq.append(y[idx * window_size: (idx + 1) * window_size])
        datas.append(data_seq)
        labels.append(label_seq)
        if len(datas) >= total_num:
            print('sufficient data loaded...')
            break
    return datas, labels


def load_data_mass(filepath, window_size, channels, total_num, normalize):
    file_names = [file for file in os.listdir(filepath) if file.endswith('-Datasub.mat')]
    file_names.sort()
    mass_channels = ['FP1', 'FP2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'Pz', 'P3', 'P4', 'T5',
                     'T6', 'Oz', 'O1', 'O2', 'EogL', 'EogR', 'Emg1', 'Emg2', 'Emg3', 'Ecg']
    channel_index = [mass_channels.index(c) for c in channels]
    datas, labels = [], []
    for file in file_names:
        print(f'loading raw data from {os.path.join(filepath, file)}')
        raw_data = sio.loadmat(os.path.join(filepath, file))
        raw_data_trans = raw_data['PSG'][:, channel_index, :]
        raw_data_trans = raw_data_trans.transpose(1, 0, 2)
        X = None
        for idx in range(raw_data_trans.shape[0]):
            series = raw_data_trans[idx]
            if normalize:
                mu, sigma = np.mean(series), np.std(series)
                series = (series - mu) / sigma
            '''
            print(f'calculating stft for channel index {idx} in mass...')
            _, _, Zxx = signal.stft(series, 100, 'hann', 256)
            Zxx = 20 * np.log10(np.abs(Zxx))
            '''
            if X is None:
                X = torch.unsqueeze(torch.tensor(series, dtype=torch.float32, requires_grad=False), dim=1)
            else:
                temp = torch.unsqueeze(torch.tensor(series, dtype=torch.float32, requires_grad=False), dim=1)
                X = torch.cat((X, temp), dim=1)
        label_name = file[:10] + '-Label.mat'
        stage_label = sio.loadmat(os.path.join(filepath, label_name))['label']
        stage_label = np.argmax(stage_label, axis=1)
        y = torch.tensor(stage_label, dtype=torch.int64, requires_grad=False)
        data_seq, label_seq, segs = [], [], X.shape[0] // window_size
        for idx in range(segs):
            data_seq.append(X[idx * window_size: (idx + 1) * window_size])
            label_seq.append(y[idx * window_size: (idx + 1) * window_size])
        datas.append(data_seq)
        labels.append(label_seq)
        if len(datas) >= total_num:
            print('sufficient data loaded...')
            break
    return datas, labels


def load_data_sleepedf(filepath, window_size, channels, total_num, normalize):
    file_names = [file for file in os.listdir(filepath)]
    file_names.sort()
    sleepedf_channels = ['Fpz-Cz', 'EOG', 'EMG']
    channel_index = [sleepedf_channels.index(c) for c in channels]
    datas, labels = [], []
    for file in file_names:
        print(f'loading raw data from {os.path.join(filepath, file)}')
        try:
            npz_file = np.load(os.path.join(filepath, file), allow_pickle=True)
        except IOError as e:
            print(f"Failed to load data from {os.path.join(filepath, file)}: {e}")
            continue
        raw_data_trans = npz_file['x'][:, :, channel_index]
        raw_data_trans = raw_data_trans.transpose(2, 0, 1)
        X = None
        for idx in range(raw_data_trans.shape[0]):
            series = raw_data_trans[idx]
            if normalize:
                mu, sigma = np.mean(series), np.std(series)
                series = (series - mu) / sigma
            '''
            print(f'calculating stft for channel index {idx} in sleepedf...')
            _, _, Zxx = signal.stft(series, 100, 'hann', 256)
            Zxx = 20 * np.log10(np.abs(Zxx))
            '''
            if X is None:
                X = torch.unsqueeze(torch.tensor(series, dtype=torch.float32, requires_grad=False), dim=1)
            else:
                temp = torch.unsqueeze(torch.tensor(series, dtype=torch.float32, requires_grad=False), dim=1)
                X = torch.cat((X, temp), dim=1)
        y = torch.tensor(npz_file['y'], dtype=torch.int64, requires_grad=False)
        data_seq, label_seq, segs = [], [], X.shape[0] // window_size
        for idx in range(segs):
            data_seq.append(X[idx * window_size: (idx + 1) * window_size])
            label_seq.append(y[idx * window_size: (idx + 1) * window_size])
        datas.append(data_seq)
        labels.append(label_seq)
        if len(datas) >= total_num:
            print('sufficient data loaded...')
            break
    return datas, labels


def load_data_physionet(filepath, window_size, channels, total_num, normalize):
    file_names = [file for file in os.listdir(filepath) if file.endswith('_x.npy')]
    file_names.sort()
    physionet_channels = ['C4', 'E1']
    channel_index = [physionet_channels.index(c) for c in channels]
    datas, labels = [], []
    for file in file_names:
        print(f'loading raw data from {os.path.join(filepath, file)}')
        try:
            npy_data = np.load(os.path.join(filepath, file))
            npy_label = np.load(os.path.join(filepath, file[:-6] + '_y.npy'))
        except IOError as e:
            print(f"Failed to load data from {os.path.join(filepath, file)}: {e}")
            continue
        raw_data_sample = npy_data[:, channel_index, :]
        raw_data_trans = raw_data_sample.transpose(1, 0, 2)
        X = None
        for idx in range(raw_data_trans.shape[0]):
            series = raw_data_trans[idx]
            if normalize:
                mu, sigma = np.mean(series), np.std(series)
                series = (series - mu) / sigma
            '''
            print(f'calculating stft for channel index {idx} in sleepedf...')
            _, _, Zxx = signal.stft(series, 100, 'hann', 256)
            Zxx = 20 * np.log10(np.abs(Zxx))
            '''
            if X is None:
                X = torch.unsqueeze(torch.tensor(series, dtype=torch.float32, requires_grad=False), dim=1)
            else:
                temp = torch.unsqueeze(torch.tensor(series, dtype=torch.float32, requires_grad=False), dim=1)
                X = torch.cat((X, temp), dim=1)
        y = torch.tensor(npy_label, dtype=torch.int64, requires_grad=False)
        data_seq, label_seq, segs = [], [], X.shape[0] // window_size
        for idx in range(segs):
            data_seq.append(X[idx * window_size: (idx + 1) * window_size])
            label_seq.append(y[idx * window_size: (idx + 1) * window_size])
        datas.append(data_seq)
        labels.append(label_seq)
        if len(datas) >= total_num:
            print('sufficient data loaded...')
            break
    return datas, labels


def load_data_hsp(filepath, window_size, channels, total_num, normalize, datadir='S0001'):
    subject_dirs = [dir_name for dir_name in os.listdir(os.path.join(filepath, datadir)) if not dir_name.startswith('.')]
    subject_dirs.sort()
    with open(os.path.join(filepath, 'S0001_final_files.txt'), 'r') as file:
        include_files = file.readlines()
        include_files = [item.strip() for item in include_files]
    exclude_files = ['sub-S0001111189848_ses-1_task-psg_eeg.fif', 'sub-S0001111190905_ses-3_task-psg_eeg.fif',
                     'sub-S0001111190905_ses-4_task-psg_eeg.fif', 'sub-S0001111191757_ses-1_task-psg_eeg.fif',
                     'sub-S0001111192352_ses-2_task-psg_eeg.fif', 'sub-S0001111192396_ses-1_task-psg_eeg.fif',
                     'sub-S0001111193501_ses-1_task-psg_eeg.fif', 'sub-S0001111193967_ses-1_task-psg_eeg.fif',
                     'sub-S0001111195626_ses-1_task-psg_eeg.fif', 'sub-S0001111198326_ses-1_task-psg_eeg.fif',
                     'sub-S0001111198446_ses-1_task-psg_eeg.fif', 'sub-S0001111198643_ses-1_task-psg_eeg.fif',
                     'sub-S0001111201541_ses-1_task-psg_eeg.fif', 'sub-S0001111204219_ses-1_task-psg_eeg.fif',
                     'sub-S0001111204219_ses-2_task-psg_eeg.fif', 'sub-S0001111204949_ses-1_task-psg_eeg.fif',
                     'sub-S0001111206862_ses-1_task-psg_annotations.csv',
                     'sub-S0001111531393_ses-2_task-psg_eeg.fif', 'sub-S0001111534168_ses-2_task-psg_eeg.fif',
                     'sub-S0001111534857_ses-2_task-psg_annotations.csv',
                     'sub-S0001111580910_ses-1_task-psg_eeg.fif', 'sub-S0001111585731_ses-1_task-psg_eeg.fif',
                     'sub-S0001111589934_ses-1_task-psg_eeg.fif', 'sub-S0001111590846_ses-1_task-psg_eeg.fif',
                     'sub-S0001111592382_ses-1_task-psg_eeg.fif', 'sub-S0001111593304_ses-1_task-psg_eeg.fif',
                     'sub-S0001111594612_ses-2_task-psg_eeg.fif', 'sub-S0001111594612_ses-6_task-psg_eeg.fif',
                     'sub-S0001111594612_ses-7_task-psg_eeg.fif', 'sub-S0001111595224_ses-2_task-psg_eeg.fif',
                     'sub-S0001111596342_ses-1_task-psg_eeg.fif', 'sub-S0001111600524_ses-1_task-psg_eeg.fif',
                     'sub-S0001111600829_ses-2_task-psg_eeg.fif', 'sub-S0001111600869_ses-1_task-psg_eeg.fif',
                     'sub-S0001111602832_ses-2_task-psg_eeg.fif', 'sub-S0001111603884_ses-1_task-psg_eeg.fif',
                     'sub-S0001111603996_ses-3_task-psg_eeg.fif', 'sub-S0001111605716_ses-1_task-psg_eeg.fif',
                     'sub-S0001111606551_ses-1_task-psg_eeg.fif', 'sub-S0001111606551_ses-2_task-psg_eeg.fif',
                     'sub-S0001111607172_ses-1_task-psg_eeg.fif', 'sub-S0001111608665_ses-3_task-psg_eeg.fif',
                     'sub-S0001111610656_ses-1_task-psg_eeg.fif', 'sub-S0001111611361_ses-1_task-psg_eeg.fif',
                     'sub-S0001111611949_ses-1_task-psg_eeg.fif', 'sub-S0001111612898_ses-1_task-psg_eeg.fif',
                     'sub-S0001111616525_ses-1_task-psg_eeg.fif',
                     'sub-S0001111617823_ses-1_task-psg_annotations.csv',
                     'sub-S0001111265591_ses-1_task-psg_eeg.fif', 'sub-S0001111325229_ses-1_task-psg_eeg.fif',
                     'sub-S0001111415453_ses-1_task-psg_eeg.fif', 'sub-S0001111418143_ses-1_task-psg_eeg.fif',
                     'sub-S0001111475060_ses-1_task-psg_eeg.fif', 'sub-S0001111480684_ses-1_task-psg_eeg.fif',
                     'sub-S0001111513011_ses-1_task-psg_eeg.fif', 'sub-S0001111520136_ses-2_task-psg_eeg.fif',
                     'sub-S0001111618995_ses-1_task-psg_eeg.fif', 'sub-S0001111236273_ses-1_task-psg_eeg.fif',
                     'sub-S0001111250016_ses-1_task-psg_eeg.fif', 'sub-S0001111255993_ses-1_task-psg_eeg.fif',
                     'sub-S0001111269982_ses-6_task-psg_eeg.fif', 'sub-S0001111281528_ses-1_task-psg_eeg.fif',
                     'sub-S0001111301802_ses-2_task-psg_eeg.fif', 'sub-S0001111303895_ses-3_task-psg_eeg.fif',
                     'sub-S0001111325116_ses-1_task-psg_eeg.fif', 'sub-S0001111343357_ses-1_task-psg_eeg.fif',
                     'sub-S0001111350330_ses-1_task-psg_eeg.fif', 'sub-S0001111382446_ses-1_task-psg_eeg.fif',
                     'sub-S0001111406573_ses-1_task-psg_eeg.fif', 'sub-S0001111414731_ses-1_task-psg_eeg.fif',
                     'sub-S0001111425041_ses-1_task-psg_eeg.fif', 'sub-S0001111447432_ses-1_task-psg_eeg.fif',
                     'sub-S0001111504311_ses-2_task-psg_eeg.fif']
    p_names = []
    for sub in subject_dirs:
        for session in os.listdir(os.path.join(filepath, datadir, sub)):
            if session.startswith('.') or os.path.join(sub, session) not in include_files:
                continue
            for data in os.listdir(os.path.join(filepath, datadir, sub, session, 'eeg')):
                if not data.startswith('.') and data.endswith('.fif') and data not in exclude_files:
                    p_names.append(os.path.join(filepath, datadir, sub, session, 'eeg', data))
    datas, labels = [], []
    for file_path in p_names:
        # print(f'loading raw data from {file_path}')
        try:
            raw_data = mne.io.read_raw_fif(file_path)
        except IOError as e:
            print(f"Failed to load data from {file_path}: {e}")
            continue
        raw_data.pick_channels(channels)
        freq = raw_data.info['sfreq']
        xall = raw_data.get_data()
        xall = signal.resample(xall, int(xall.shape[1] * 100 // freq), axis=1)
        label_file = file_path.replace('_eeg.fif', '_annotations.csv')
        if not os.path.exists(label_file):
            print(f"different label file: {label_file}")
            break
        df = pd.read_csv(label_file)
        df_stage = df[df['event'].str.startswith('Sleep_stage_', na=False)]
        df_valid = df_stage[df_stage['event'] != 'Sleep_stage_?'].copy()
        valid_start_idx = df_valid.index[0]
        valid_end_idx = df_valid.index[-1]
        df_sleep = df_stage.loc[valid_start_idx:valid_end_idx].reset_index(drop=True)
        stage2int = {
            'Sleep_stage_W': 0,
            'Sleep_stage_N1': 1,
            'Sleep_stage_N2': 2,
            'Sleep_stage_N3': 3,
            'Sleep_stage_R': 4,
            'Sleep_stage_REM': 4
        }
        df_sleep['label'] = df_sleep['event'].map(stage2int)
        y_group = df_sleep['label'].to_numpy()
        if np.isnan(y_group).any():
            print("y_group NaN file exists：", file_path)
            print("NaN in label：", df_sleep[df_sleep['label'].isnull()])
            continue
        start_epoch = int(df_sleep.iloc[0]['epoch'])
        end_epoch = int(df_sleep.iloc[-1]['epoch'])
        samples_per_epoch = 3000
        start_sample = (start_epoch - 1) * samples_per_epoch
        end_sample = end_epoch * samples_per_epoch
        xall = xall[:, start_sample:end_sample]
        if normalize:
            xall = (xall - np.mean(xall, axis=1, keepdims=True)) / np.std(xall, axis=1, keepdims=True)
        num_channels = xall.shape[0]
        xall = xall.reshape(num_channels, -1, 3000).transpose(1, 0, 2)
        X = torch.tensor(xall, dtype=torch.float32, requires_grad=False)
        y = torch.tensor(y_group, dtype=torch.int64, requires_grad=False)
        data_seq, label_seq, segs = [], [], X.shape[0] // window_size
        for idx in range(segs):
            data_seq.append(X[idx * window_size: (idx + 1) * window_size])
            label_seq.append(y[idx * window_size: (idx + 1) * window_size])
        datas.append(data_seq)
        labels.append(label_seq)
        if len(datas) >= total_num:
            print('sufficient data loaded...')
            break
    return datas, labels


class DataWrapper(Dataset):
    def __init__(self, data, label, args, augmentation=False, task=None):
        assert len(data) == len(label)
        self.data = data
        self.label = label
        self.task = task
        self.args = args
        self.augmentation = augmentation

    def __getitem__(self, item):
        data, label = self.data[item], self.label[item]
        if self.augmentation and random.random() < self.args.time_reverse_rate:
            data = torch.flip(data, dims=[-1])
        if self.task is None:
            return data, label
        else:
            return data, label, self.task[item]

    def __len__(self):
        return len(self.data)


def create_fold_monolithic(train, valid, test, datas_tasklist, labels_tasklist, args):
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    cnt = 0
    datas_selected, labels_selected = [], []
    for datas, labels in zip(datas_tasklist, labels_tasklist):
        for idx in train[cnt]:
            for X, y in zip(datas[idx], labels[idx]):
                datas_selected.append(X)
                labels_selected.append(y)
        cnt += 1
    train_datasets.append(DataWrapper(datas_selected, labels_selected, args, True))
    cnt = 0
    datas_selected, labels_selected = [], []
    for datas, labels in zip(datas_tasklist, labels_tasklist):
        for idx in valid[cnt]:
            for X, y in zip(datas[idx], labels[idx]):
                datas_selected.append(X)
                labels_selected.append(y)
        cnt += 1
    valid_datasets.append(DataWrapper(datas_selected, labels_selected, args))
    cnt = 0
    datas_selected, labels_selected = [], []
    for datas, labels in zip(datas_tasklist, labels_tasklist):
        for idx in test[cnt]:
            for X, y in zip(datas[idx], labels[idx]):
                datas_selected.append(X)
                labels_selected.append(y)
        cnt += 1
    test_datasets.append(DataWrapper(datas_selected, labels_selected, args))
    return train_datasets, valid_datasets, test_datasets


def create_fold_task_separated(train, valid, test, datas_tasklist, labels_tasklist, args):
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    cnt = 0
    for datas, labels in zip(datas_tasklist, labels_tasklist):
        datas_selected, labels_selected = [], []
        for idx in train[cnt]:
            for X, y in zip(datas[idx], labels[idx]):
                datas_selected.append(X)
                labels_selected.append(y)
        train_datasets.append(DataWrapper(datas_selected, labels_selected, args, True))
        datas_selected, labels_selected = [], []
        for idx in valid[cnt]:
            for X, y in zip(datas[idx], labels[idx]):
                datas_selected.append(X)
                labels_selected.append(y)
        valid_datasets.append(DataWrapper(datas_selected, labels_selected, args))
        datas_selected, labels_selected = [], []
        for idx in test[cnt]:
            for X, y in zip(datas[idx], labels[idx]):
                datas_selected.append(X)
                labels_selected.append(y)
        test_datasets.append(DataWrapper(datas_selected, labels_selected, args))
        cnt += 1
    return train_datasets, valid_datasets, test_datasets


def load_all_datasets(args):
    datas, labels = [], []
    for task_name in args.task_names:
        if task_name == 'ISRUC1':
            normalize = args.normalize
            file_path = os.path.join(args.path_prefix, args.isruc1_path)
            task_data, task_label = load_data_isruc1(file_path, args.window_size, args.isruc1,
                                                     args.total_num['ISRUC1'], normalize)
            datas.append(task_data)
            labels.append(task_label)
        elif task_name == 'SHHS':
            normalize = args.normalize
            file_path = os.path.join(args.path_prefix, args.shhs_path)
            task_data, task_label = load_data_shhs(file_path, args.window_size, args.shhs,
                                                   args.total_num['SHHS'], normalize)
            datas.append(task_data)
            labels.append(task_label)
        elif task_name == 'MASS':
            normalize = args.normalize
            file_path = os.path.join(args.path_prefix, args.mass_path)
            task_data, task_label = load_data_mass(file_path, args.window_size, args.mass,
                                                   args.total_num['MASS'], normalize)
            datas.append(task_data)
            labels.append(task_label)
        elif task_name == 'Sleep-EDF':
            normalize = args.normalize
            file_path = os.path.join(args.path_prefix, args.sleep_edf_path)
            task_data, task_label = load_data_sleepedf(file_path, args.window_size, args.sleep_edf,
                                                       args.total_num['Sleep-EDF'], normalize)
            datas.append(task_data)
            labels.append(task_label)
        elif task_name == 'PhysioNet':
            normalize = args.normalize
            file_path = os.path.join(args.path_prefix, args.physionet_path)
            task_data, task_label = load_data_physionet(file_path, args.window_size, args.physionet,
                                                        args.total_num['PhysioNet'], normalize)
            datas.append(task_data)
            labels.append(task_label)
        elif task_name == 'HSP':
            normalize = args.normalize
            file_path = os.path.join(args.path_prefix, args.hsp_path)
            task_data, task_label = load_data_hsp(file_path, args.window_size, args.hsp,
                                                  args.total_num['HSP'], normalize)
            datas.append(task_data)
            labels.append(task_label)
    return datas, labels


if __name__ == '__main__':
    '''
    datas, labels = load_data_sleepedf('/home/ShareData/sleep-edf-153-3chs', 5, ['Fpz-Cz', 'EOG'], 5)
    train, valid, test = create_fold([0, 1, 2], [3], [4], [datas, datas, datas], [labels, labels, labels])
    train_loader = DataLoader(train, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid, batch_size=8, shuffle=False)
    test_loader = DataLoader(test, batch_size=8, shuffle=False)
    print('train loader...')
    for X, y in train_loader:
        print(f'{X.shape}, {y.shape}')
    print('valid loader...')
    for X, y in valid_loader:
        print(f'{X.shape}, {y.shape}')
    print('test loader...')
    for X, y in test_loader:
        print(f'{X.shape}, {y.shape}')
    datas, labels = load_data_physionet('/root/autodl-tmp/PhysioNet-Challenge-2018_sub251_C4E1', 10, ['C4', 'E1'], 5, True)
    '''
    load_data_hsp('/root/autodl-tmp/HSP_processed_0624_taiyang', 10, ['C4', 'E1'], 5, True)
