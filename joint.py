import argparse
from train import *
from main import args

args.task_num = 1
args.task_names = ['Joint']
args.replay_mode = 'none'
total_num = 0
for key, value in args.total_num.items():
    total_num += value
args.total_num['Joint'] = total_num

if __name__ == '__main__':
    R, exp_log = train_k_fold(args)
    write_format(R, args, 'cl_output_record_' + 'joint' + '.txt', exp_log)
    exp_log.write()
