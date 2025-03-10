import argparse
from train import *

parser = argparse.ArgumentParser(description='experiment settings')
parser.add_argument('--path_prefix', type=str, nargs='?', default='/root/autodl-tmp', help='path to datasets')
parser.add_argument('--random_seed', type=int, nargs='?', default=42, help='random seed')
parser.add_argument('--isruc1_path', type=str, nargs='?',
                    default='ISRUC-1', help='file path of isruc1 dataset')
parser.add_argument('--isruc1', nargs='+', default=['C4_A1', 'LOC_A2'], help='channels for isruc1')
parser.add_argument('--shhs_path', type=str, nargs='?',
                    default='shhs1_process6', help='file path of shhs dataset')
parser.add_argument('--shhs', nargs='+', default=['EEG', 'EOG(L)'], help='channels for shhs')
parser.add_argument('--mass_path', type=str, nargs='?',
                    default='MASS_SS3_3000_25C-Cz', help='file path of mass dataset')
parser.add_argument('--mass', nargs='+', default=['C4', 'EogL'], help='channels for mass')
parser.add_argument('--sleep_edf_path', type=str, nargs='?',
                    default='sleep-edf-153-3chs', help='file path of sleepedf dataset')
parser.add_argument('--sleep_edf', nargs='+', default=['Fpz-Cz', 'EOG'], help='channels of sleepedf')
parser.add_argument('--normalize', type=bool, nargs='?', default=True, help='whether normalize samples in subjects')
parser.add_argument('--task_num', type=int, nargs='?', default=4, help='number of tasks')
parser.add_argument('--task_names', nargs='+', default=['ISRUC1', 'SHHS', 'MASS', 'Sleep-EDF'],
                    help='the list of task names')
parser.add_argument('--cuda_idx', type=int, nargs='?', default=0, help='device index')
parser.add_argument('--window_size', type=int, nargs='?', default=10, help='length of sequence')
parser.add_argument('--total_num', nargs='+', default={'ISRUC1': 100, 'SHHS': 200, 'MASS': 60, 'Sleep-EDF': 150},
                    help='number of examples for each task')
parser.add_argument('--fold_num', type=int, nargs='?', default=10, help='number of a single fold')
parser.add_argument('--num_epochs', type=int, nargs='?', default=200, help='number of epochs')
parser.add_argument('--batch_size', type=int, nargs='?', default=32, help='batch size')
parser.add_argument('--valid_epoch', type=int, nargs='?', default=5, help='validating interval')
parser.add_argument('--valid_batch', type=int, nargs='?', default=32, help='validating batch size')
parser.add_argument('--dropout', type=float, nargs='?', default=0.1, help='drop out ratio')
parser.add_argument('--weight_decay', type=float, nargs='?', default=1e-4, help='weight decay value')
parser.add_argument('--lr', type=float, nargs='?', default=1e-4, help='learning rate')
parser.add_argument('--replay_mode', type=str, nargs='?', default='none', help='continual learning strategy')
parser.add_argument('--min_epoch', type=float, nargs='?', default=10, help='min epochs for model saving')
'''knowledge distillation setting'''
parser.add_argument('--tau', type=float, nargs='?', default=1, help='temperature for knowledge distillation')
'''generative replay settings'''
parser.add_argument('--num_epochs_generator', type=int, nargs='?', default=100, help='number of epochs for generator')
parser.add_argument('--lr_seq_gen', type=float, nargs='?', default=1e-4, help='learning rate for sequential generator')
parser.add_argument('--beta', type=float, nargs='?', default=0.1, help='coefficient of kl loss')
parser.add_argument('--alpha', type=float, nargs='?', default=10, help='coefficient of replay loss')
parser.add_argument('--gamma', type=float, nargs='?', default=1e-2, help='updating rate for running loss')
'''ewc settings'''
parser.add_argument('--ewc_lambda', type=float, nargs='?', default=1e3, help='coefficient for ewc penalty')
parser.add_argument('--ewc_gamma', type=float, nargs='?', default=0.4, help='updating rate for FIM')
parser.add_argument('--ewc_batches', type=int, nargs='?', default=256, help='number of batches for calculating FIM')
'''der settings'''
parser.add_argument('--der_alpha', type=float, nargs='?', default=0.5, help='dark experience alpha')
parser.add_argument('--der_beta', type=float, nargs='?', default=0.5, help='dark experience beta')
'''dt2w settings'''
parser.add_argument('--dtw_lambda', type=float, nargs='?', default=0.03, help='coefficient for dtw loss')
'''data augmentation settings'''
parser.add_argument('--time_reverse_rate', type=float, nargs='?', default=0.1, help='frequency for time reversion')
'''other settings'''
parser.add_argument('--enable_multihead', action='store_true', help='whether enable multihead')
parser.add_argument('--replay_buffer', type=float, nargs='?', default=384, help='replay buffer size')
parser.add_argument('--joint_training', action='store_true', help='start joint training')
args = parser.parse_args()

if __name__ == '__main__':
    R, exp_log = train_k_fold(args)
    write_format(R, args, 'cl_output_record_' + args.replay_mode + '.txt', exp_log)
    exp_log.write()
    exp_log.save_params()
