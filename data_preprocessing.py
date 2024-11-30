import numpy as np
import pickle as pkl
import scipy.io as sio
import torch
import os
from scipy import signal
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def load_data_isruc1(filepath, window_size, channels, total_num):
    file_names = [file for file in os.listdir(filepath) if file.endswith('.mat')]
    file_names.sort()
    datas, labels = [], []
    for file in file_names:
        print(f'loading raw data from {os.path.join(filepath, file)}...')
        raw_data = sio.loadmat(os.path.join(filepath, file))
        X = None
        for channel in channels:
            data_resampled = signal.resample(raw_data[channel], 3000, axis=1)
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
        assert window_size % 2 == 1, 'window size must be an odd number'
        data_seq, label_seq, context = [], [], window_size // 2
        padding = torch.zeros((context, X.shape[1], X.shape[2]), dtype=torch.float32, requires_grad=False)
        X = torch.cat((padding, X, padding), dim=0)
        for idx in range(context, X.shape[0] - context):
            data_seq.append(X[idx - context: idx + context + 1])
            label_seq.append(y[idx - context])
        datas.append(data_seq)
        labels.append(label_seq)
        if len(datas) >= total_num:
            print('sufficient data loaded...')
            break
    return datas, labels


def load_data_shhs(filepath, window_size, channels, total_num):
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
        assert window_size % 2 == 1, 'window size must be an odd number'
        data_seq, label_seq, context = [], [], window_size // 2
        padding = torch.zeros((context, X.shape[1], X.shape[2]), dtype=torch.float32, requires_grad=False)
        X = torch.cat((padding, X, padding), dim=0)
        for idx in range(context, X.shape[0] - context):
            data_seq.append(X[idx - context: idx + context + 1])
            label_seq.append(y[idx - context])
        datas.append(data_seq)
        labels.append(label_seq)
        if len(datas) >= total_num:
            print('sufficient data loaded...')
            break
    return datas, labels


def load_data_mass(filepath, window_size, channels, total_num):
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
        assert window_size % 2 == 1, 'window size must be an odd number'
        data_seq, label_seq, context = [], [], window_size // 2
        padding = torch.zeros((context, X.shape[1], X.shape[2]), dtype=torch.float32, requires_grad=False)
        X = torch.cat((padding, X, padding), dim=0)
        for idx in range(context, X.shape[0] - context):
            data_seq.append(X[idx - context: idx + context + 1])
            label_seq.append(y[idx - context])
        datas.append(data_seq)
        labels.append(label_seq)
        if len(datas) >= total_num:
            print('sufficient data loaded...')
            break
    return datas, labels


def load_data_sleepedf(filepath, window_size, channels, total_num):
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
        assert window_size % 2 == 1, 'window size must be an odd number'
        data_seq, label_seq, context = [], [], window_size // 2
        padding = torch.zeros((context, X.shape[1], X.shape[2]), dtype=torch.float32, requires_grad=False)
        X = torch.cat((padding, X, padding), dim=0)
        for idx in range(context, X.shape[0] - context):
            data_seq.append(X[idx - context: idx + context + 1])
            label_seq.append(y[idx - context])
        datas.append(data_seq)
        labels.append(label_seq)
        if len(datas) >= total_num:
            print('sufficient data loaded...')
            break
    return datas, labels


class DataWrapper(Dataset):
    def __init__(self, data, label, task=None):
        assert len(data) == len(label)
        self.data = data
        self.label = label
        self.task = task

    def __getitem__(self, item):
        if self.task is None:
            return self.data[item], self.label[item]
        else:
            return self.data[item], self.label[item], self.task[item]

    def __len__(self):
        return len(self.data)


def create_fold(train, valid, test, datas_tasklist, labels_tasklist):
    train_data, train_label, train_task = [], [], []
    valid_data, valid_label, valid_task = [], [], []
    test_data, test_label, test_task = [], [], []
    t = 0
    for datas, labels in zip(datas_tasklist, labels_tasklist):
        for idx in train:
            for X, y in zip(datas[idx], labels[idx]):
                train_data.append(X)
                train_label.append(y)
                train_task.append(t)
        for idx in valid:
            for X, y in zip(datas[idx], labels[idx]):
                valid_data.append(X)
                valid_label.append(y)
                valid_task.append(t)
        for idx in test:
            for X, y in zip(datas[idx], labels[idx]):
                test_data.append(X)
                test_label.append(y)
                test_task.append(t)
        t += 1
    train_dataset = DataWrapper(train_data, train_label)
    valid_dataset = DataWrapper(valid_data, valid_label)
    test_dataset = DataWrapper(test_data, test_label)
    return train_dataset, valid_dataset, test_dataset


def create_fold_task_separated(train, valid, test, datas_tasklist, labels_tasklist):
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
        train_datasets.append(DataWrapper(datas_selected, labels_selected))
        datas_selected, labels_selected = [], []
        for idx in valid[cnt]:
            for X, y in zip(datas[idx], labels[idx]):
                datas_selected.append(X)
                labels_selected.append(y)
        valid_datasets.append(DataWrapper(datas_selected, labels_selected))
        datas_selected, labels_selected = [], []
        for idx in test[cnt]:
            for X, y in zip(datas[idx], labels[idx]):
                datas_selected.append(X)
                labels_selected.append(y)
        test_datasets.append(DataWrapper(datas_selected, labels_selected))
        cnt += 1
    return train_datasets, valid_datasets, test_datasets


def load_all_datasets(args):
    datas, labels = [], []
    for task_name in args.task_names:
        if task_name == 'ISRUC1':
            task_data, task_label = load_data_isruc1(args.isruc1_path, args.window_size, args.isruc1, args.total_num['ISRUC1'])
            datas.append(task_data)
            labels.append(task_label)
        elif task_name == 'SHHS':
            task_data, task_label = load_data_shhs(args.shhs_path, args.window_size, args.shhs, args.total_num['SHHS'])
            datas.append(task_data)
            labels.append(task_label)
        elif task_name == 'MASS':
            task_data, task_label = load_data_mass(args.mass_path, args.window_size, args.mass, args.total_num['MASS'])
            datas.append(task_data)
            labels.append(task_label)
        elif task_name == 'Sleep-EDF':
            task_data, task_label = load_data_sleepedf(args.sleep_edf_path, args.window_size, args.sleep_edf, args.total_num['Sleep-EDF'])
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
    '''
    pass
