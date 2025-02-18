import random
import json
import os
from main import args
from datetime import datetime


class LogDocument:
    def __init__(self, args):
        self.assignment_idx = random.randint(100, 256)
        self.file_path = (str(args.replay_mode) + f'_experiment{self.assignment_idx}_' +
                          datetime.now().strftime("%Y-%m-%d") + '.json')
        self.all_information = {
            'log_startint_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'exp_assignment_index': self.assignment_idx,
            'exp_args': vars(args),
            'performance': {},
            'train_info': {}
        }

    def append(self, fields, content):
        pointer = self.all_information
        for idx in range(len(fields)):
            if idx + 1 != len(fields):
                if fields[idx] not in pointer:
                    pointer[fields[idx]] = {}
                pointer = pointer[fields[idx]]
            else:
                pointer[fields[idx]] = content

    def update_test_results(self, test_results, fold_idx):
        cnt = 0
        for task_accs, task_mF1s in test_results:
            self.append(['performance', f'acc_on_task{cnt}_fold{fold_idx}'], task_accs)
            self.append(['performance', f'mF1_on_task{cnt}_fold{fold_idx}'], task_mF1s)
            cnt += 1

    def write(self):
        with open(os.path.join('results', self.file_path), 'w', encoding='utf-8') as file:
            json.dump(self.all_information, file, indent=4)


if __name__ == '__main__':
    log = LogDocument(args)
    log.write()
