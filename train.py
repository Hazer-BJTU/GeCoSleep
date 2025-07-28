import sys
import random
import numpy as np
from clnetworks import *
from GeCoSleep import *
from baselines import *
from data_preprocessing import *
from logs import *
from BayesEEGNet import *
from DeepSleepNet import *


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_cl(args, trains, valids, tests, fold_idx, logs):
    test_results = []
    count_results = []
    clnetwork = None
    if args.replay_mode == 'none':
        clnetwork = CLnetwork(args, fold_idx, logs)
    elif args.replay_mode == 'generative':
        clnetwork = EEGGRnetwork(args, fold_idx, logs)
    elif args.replay_mode == 'fine_tuning':
        clnetwork = FineTuning(args, fold_idx, logs)
    elif args.replay_mode == 'independent':
        clnetwork = Independent(args, fold_idx, logs)
    elif args.replay_mode == 'experience':
        clnetwork = ExperienceReplay(args, fold_idx, logs)
    elif args.replay_mode == 'lwf':
        clnetwork = LwFnetwork(args, fold_idx, logs)
    elif args.replay_mode == 'ewc':
        clnetwork = EWCnetwork(args, fold_idx, logs)
    elif args.replay_mode == 'der':
        clnetwork = DERnetwork(args, fold_idx, logs)
    elif args.replay_mode == 'dtw':
        clnetwork = DTWnetwork(args, fold_idx, logs)
    elif args.replay_mode == 'tagem':
        clnetwork = TAGEMnetwork(args, fold_idx, logs)
    elif args.replay_mode == 'agem':
        clnetwork = AGEM(args, fold_idx, logs)
    elif args.replay_mode == 'bayes':
        clnetwork = BayesCLNetwork(args, fold_idx, logs)
    elif args.replay_mode == 'deep':
        clnetwork = DeepCLNetwork(args, fold_idx, logs)
    confusion = ConfusionMatrix(args.task_num)
    print('start first testing...')
    if args.replay_mode == 'packnet':
        confusion = evaluate_tasks_packnet(clnetwork.net, tests, confusion, clnetwork.device, clnetwork, args.valid_batch)
    elif args.replay_mode == 'lwf' or args.replay_mode == 'generative':
        confusion = evaluate_tasks_multihead(clnetwork.net, tests, confusion, clnetwork.device, args.valid_batch)
    elif args.replay_mode == 'bayes':
        confusion = evaluate_tasks_bayes(clnetwork.net, tests, confusion, clnetwork.device, args.valid_batch)
    else:
        confusion = evaluate_tasks(clnetwork.net, tests, confusion, clnetwork.device, args.valid_batch)
    test_results.append((confusion.accuracy(keep_list=True), confusion.macro_f1(keep_list=True)))
    count_results.append(confusion.get_matrix())
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
        clnetwork.end_task(trains[task_idx])
        confusion.clear()
        print(f'start testing...')
        if args.replay_mode == 'packnet':
            bestnet = SleepNet(len(args.isruc1), args.dropout)
            bestnet.load_state_dict(torch.load(clnetwork.best_net_memory[task_idx], weights_only=True))
            confusion = evaluate_tasks_packnet(bestnet, tests, confusion, clnetwork.device, clnetwork, args.valid_batch)
        elif args.replay_mode == 'lwf' or args.replay_mode == 'generative':
            bestnet = MultiHeadSleepNet(len(args.isruc1), args.dropout, args.task_num, args.enable_multihead)
            bestnet.load_state_dict(torch.load(clnetwork.best_net_memory[task_idx], weights_only=True))
            confusion = evaluate_tasks_multihead(bestnet, tests, confusion, clnetwork.device, args.valid_batch)
        elif args.replay_mode == 'bayes':
            params = args.bayes_eeg_params
            bestnet = BayesEEGNet(
                hidden_size=params['hiddenDim'],
                output_size=params['targetDim'],
                graph_node_dim=params['graph_dim'],
                num_nodes=params['num_nodes'],
                last_dense=params['dense']
            )
            bestnet.load_state_dict(torch.load(clnetwork.best_net_memory[task_idx], weights_only=True))
            confusion = evaluate_tasks_bayes(bestnet, tests, confusion, clnetwork.device, args.valid_batch)
        elif args.replay_mode == 'deep':
            bestnet = DeepSleepNet(len(args.isruc1), dropout=args.dropout)
            bestnet.load_state_dict(torch.load(clnetwork.best_net_memory[task_idx], weights_only=True))
            confusion = evaluate_tasks(bestnet, tests, confusion, clnetwork.device, args.valid_batch)
        else:
            bestnet = SleepNet(len(args.isruc1), args.dropout)
            bestnet.load_state_dict(torch.load(clnetwork.best_net_memory[task_idx], weights_only=True))
            confusion = evaluate_tasks(bestnet, tests, confusion, clnetwork.device, args.valid_batch)
        test_results.append((confusion.accuracy(keep_list=True), confusion.macro_f1(keep_list=True)))
        count_results.append(confusion.get_matrix())
    return test_results, count_results


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
    exp_log = LogDocument(args)
    set_random_seed(args.random_seed)
    fold_task_test_idx, fold_task_valid_idx, fold_task_train_idx = allocate_fold(args)
    datas, labels = load_all_datasets(args)
    if args.joint_training:
        args.task_num = 1
    total_results = torch.zeros((args.task_num + 1, args.task_num, 2), dtype=torch.float32, requires_grad=False)
    for fold_idx in range(len(fold_task_test_idx)):
        if args.joint_training:
            trains, valids, tests = create_fold_monolithic(
                fold_task_train_idx[fold_idx],
                fold_task_valid_idx[fold_idx],
                fold_task_test_idx[fold_idx],
                datas,
                labels,
                args
            )
        else:
            trains, valids, tests = create_fold_task_separated(
                fold_task_train_idx[fold_idx],
                fold_task_valid_idx[fold_idx],
                fold_task_test_idx[fold_idx],
                datas,
                labels,
                args
            )
        print(f'start fold {fold_idx}:')
        test_results, count_results = train_cl(args, trains, valids, tests, fold_idx, exp_log)
        exp_log.update_test_results(test_results, count_results, fold_idx)
        exp_log.write()
        for i in range(args.task_num + 1):
            for j in range(args.task_num):
                total_results[i][j][0] += test_results[i][0][j] / len(fold_task_test_idx)
                total_results[i][j][1] += test_results[i][1][j] / len(fold_task_test_idx)
    return total_results, exp_log


def write_format(R, args, filepath='cl_output_record.txt', logs=None):
    original_stdout = sys.stdout
    prefix = os.path.join('results', logs.dir_path)
    with open(os.path.join(prefix, filepath), 'w') as file:
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
        bestacc, bestmf1 = 0, 0
        aacc, bwt, fwt = 0, 0, 0
        af1, bwtf1, fwtf1 = 0, 0, 0
        for j in range(args.task_num):
            bestacc += R[j + 1][j][0]
            bestmf1 += R[j + 1][j][1]
            aacc += R[args.task_num][j][0]
            af1 += R[args.task_num][j][1]
            if j != args.task_num - 1:
                bwt += R[args.task_num][j][0] - R[j + 1][j][0]
                bwtf1 += R[args.task_num][j][1] - R[j + 1][j][1]
            if j != 0:
                fwt += R[j][j][0] - R[0][j][0]
                fwtf1 += R[j][j][1] - R[0][j][1]
        bestacc, bestmf1 = bestacc / args.task_num, bestmf1 / args.task_num
        aacc, bwt, fwt = aacc / args.task_num, bwt / max(args.task_num - 1, 1), fwt / max(args.task_num - 1, 1)
        af1, bwtf1, fwtf1 = af1 / args.task_num, bwtf1 / max(args.task_num - 1, 1), fwtf1 / max(args.task_num - 1, 1)
        print(f'average acc: {aacc:.3f}, average macro F1: {af1:.3f}')
        print(f'best acc: {bestacc:.3f}, best macro F1: {bestmf1:.3f}')
        print(f'BWT: {bwt:.3f}, BWT(mF1): {bwtf1:.3f}')
        print(f'FWT: {fwt:.3f}, FWT(mF1): {fwtf1:.3f}')
        if logs is not None:
            def to_float(x):
                return x if isinstance(x, float) else x.item()
            logs.append(['performance', 'average'], {'acc': to_float(aacc), 'mF1': to_float(af1)})
            logs.append(['performance', 'best'], {'acc': to_float(bestacc), 'mF1': to_float(bestmf1)})
            logs.append(['performance', 'BWT'], {'acc': to_float(bwt), 'mF1': to_float(bwtf1)})
            logs.append(['performance', 'FWT'], {'acc': to_float(fwt), 'mF1': to_float(fwtf1)})
    sys.stdout = original_stdout


if __name__ == '__main__':
    pass
