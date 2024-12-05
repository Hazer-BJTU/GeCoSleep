from clnetworks import *
from models import *
from torch.utils.data import DataLoader
from data_preprocessing import *
from EEGGR import *
import sys


def train_cl(args, trains, valids, tests):
    test_results = []
    clnetwork = None
    if args.replay_mode == 'none':
        clnetwork = CLnetwork(args)
    elif args.replay_mode == 'generative':
        clnetwork = EEGGRnetwork(args)
    confusion = ConfusionMatrix(args.task_num)
    print('start first testing...')
    confusion = evaluate_tasks(clnetwork.net, tests, confusion, clnetwork.device, args.valid_batch)
    test_results.append((confusion.accuracy(), confusion.macro_f1()))
    for task_idx in range(args.task_num):
        print(f'start task {task_idx}:')
        clnetwork.start_task()
        train_loader = DataLoader(trains[task_idx], args.batch_size, True)
        print(f'number of samples: {len(trains[task_idx])}')
        for epoch in range(args.num_epochs):
            clnetwork.start_epoch()
            for X, y in train_loader:
                if epoch == 0:
                    clnetwork.observe(X, y, True)
                else:
                    clnetwork.observe(X, y, False)
            clnetwork.end_epoch(valids[task_idx])
        clnetwork.end_task()
        confusion.clear()
        print(f'start testing...')
        bestnet = SleepNet(2, args.dropout)
        bestnet.load_state_dict(torch.load(clnetwork.best_net_memory[task_idx], weights_only=True))
        if args.replay_mode == 'packnet':
            confusion = evaluate_tasks_packnet(bestnet, tests, confusion, clnetwork.device, clnetwork, args.valid_batch)
        else:
            confusion = evaluate_tasks(bestnet, tests, confusion, clnetwork.device, args.valid_batch)
        test_results.append((confusion.accuracy(), confusion.macro_f1()))
    return test_results


def allocate_fold(args):
    assert args.fold_num > 2
    fold_task_test_idx = []
    fold_task_valid_idx = []
    fold_task_train_idx = []
    for fold_idx in range(args.fold_num):
        task_test_idx = []
        task_valid_idx = []
        task_train_idx = []
        for task_name in args.task_names:
            num_samples = args.total_num[task_name]
            assert num_samples % args.fold_num == 0
            num_samples_fold = num_samples // args.fold_num
            task_test_idx.append([i for i in range(fold_idx * num_samples_fold, (fold_idx + 1) * num_samples_fold)])
            task_valid_idx.append([(i % num_samples) for i in range((fold_idx + 1) * num_samples_fold, (fold_idx + 2) * num_samples_fold)])
            task_train_idx.append([i for i in range(num_samples) if (i not in task_test_idx[-1] and i not in task_valid_idx[-1])])
        fold_task_test_idx.append(task_test_idx)
        fold_task_valid_idx.append(task_valid_idx)
        fold_task_train_idx.append(task_train_idx)
    return fold_task_test_idx, fold_task_valid_idx, fold_task_train_idx


def train_k_fold(args):
    fold_task_test_idx, fold_task_valid_idx, fold_task_train_idx = allocate_fold(args)
    datas, labels = load_all_datasets(args)
    total_results = torch.zeros((args.task_num + 1, args.task_num, 2), dtype=torch.float32, requires_grad=False)
    for fold_idx in range(len(fold_task_test_idx)):
        trains, valids, tests = create_fold_task_separated(fold_task_train_idx[fold_idx], fold_task_valid_idx[fold_idx], fold_task_test_idx[fold_idx], datas, labels)
        print(f'start fold {fold_idx}:')
        test_results = train_cl(args, trains, valids, tests)
        for i in range(args.task_num + 1):
            for j in range(args.task_num):
                total_results[i][j][0] += test_results[i][0][j] / len(fold_task_test_idx)
                total_results[i][j][1] += test_results[i][1][j] / len(fold_task_test_idx)
    return total_results


def write_format(R, args, filepath='cl_output_record.txt'):
    original_stdout = sys.stdout
    with open(filepath, 'w') as file:
        sys.stdout = file
        print('tasks: ', end='')
        for i in range(args.task_num):
            print(f'     [{args.task_names[i]}]   ', end=' ')
        print('   [AVG]   ')
        print('-' * (16 * args.task_num + 24))
        for i in range(args.task_num + 1):
            avg_acc, avg_f1 = 0.0, 0.0
            print(f'task:{i} |', end='')
            for j in range(args.task_num):
                print(f' {R[i][j][0]:.3f} / {R[i][j][1]:.3f} ', end='|')
                avg_acc += R[i][j][0] / args.task_num
                avg_f1 += R[i][j][1] / args.task_num
            print(f' {avg_acc:.3f} / {avg_f1:.3f} |')
        print('-' * (16 * args.task_num + 24))
        aacc, bwt, fwt = 0, 0, 0
        af1, bwtf1, fwtf1 = 0, 0, 0
        for j in range(args.task_num):
            aacc += R[args.task_num][j][0]
            af1 += R[args.task_num][j][1]
            if j != args.task_num - 1:
                bwt += R[args.task_num][j][0] - R[j + 1][j][0]
                bwtf1 += R[args.task_num][j][1] - R[j + 1][j][1]
            if j != 0:
                fwt += R[j][j][0] - R[0][j][0]
                fwtf1 += R[j][j][1] - R[0][j][1]
        aacc, bwt, fwt = aacc / args.task_num, bwt / (args.task_num - 1), fwt / (args.task_num - 1)
        af1, bwtf1, fwtf1 = af1 / args.task_num, bwtf1 / (args.task_num - 1), fwtf1 / (args.task_num - 1)
        print(f'average acc: {aacc:.3f}, average macro F1: {af1:.3f}')
        print(f'BWT: {bwt:.3f}, BWT(mF1): {bwtf1:.3f}')
        print(f'FWT: {fwt:.3f}, FWT(mF1): {fwtf1:.3f}')
    sys.stdout = original_stdout


if __name__ == '__main__':
    pass
