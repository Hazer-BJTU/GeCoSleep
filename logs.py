import random
import json
import os
from datetime import datetime


class LogDocument:
    def __init__(self, args):
        self.args = args
        self.assignment_idx = None
        self.file_path = None
        self.dir_path = None
        while True:
            if self.try_assign_filename():
                break
        self.all_information = {
            'log_starting_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'log_ending_time': '',
            'exp_assignment_index': self.assignment_idx,
            'exp_args': vars(args),
            'performance': {},
            'train_info': {}
        }

    def try_assign_filename(self):
        self.assignment_idx = random.randint(1000, 9999)
        if self.args.joint_training:
            self.file_path = ('joint' + f'_experiment{self.assignment_idx}_' +
                              datetime.now().strftime("%Y-%m-%d") + '.json')
        else:
            self.file_path = (str(self.args.replay_mode) + f'_experiment{self.assignment_idx}_' +
                              datetime.now().strftime("%Y-%m-%d") + '.json')
        self.dir_path = self.file_path[:-5]
        return not os.path.exists(os.path.join('results', self.dir_path))

    def append(self, fields, content):
        pointer = self.all_information
        for idx in range(len(fields)):
            if idx + 1 != len(fields):
                if fields[idx] not in pointer:
                    pointer[fields[idx]] = {}
                pointer = pointer[fields[idx]]
            else:
                pointer[fields[idx]] = content

    def update_test_results(self, test_results, count_results, fold_idx):
        cnt = 0
        for task_accs, task_mF1s in test_results:
            self.append(['performance', f'task{cnt}_fold{fold_idx}'], {'acc': task_accs, 'mF1': task_mF1s})
            cnt += 1
        cnt = 0
        for matrices in count_results:
            self.append(['performance', f'task{cnt}_fold{fold_idx}', 'confusion_matrices'], matrices)
            cnt += 1

    def write(self):
        self.all_information['log_ending_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = os.path.join('results', self.dir_path)
        if not os.path.exists(prefix):
            os.system(f'mkdir -p {prefix}')
        with open(os.path.join(prefix, self.file_path), 'w', encoding='utf-8') as file:
            json.dump(self.all_information, file, indent=4)

    def save_params(self):
        prefix = os.path.join('results', self.dir_path)
        print('packaging models...')
        os.system(
            f'find modelsaved -type f -name "*.pth" -print0 | '
            f"tar --null -czvf {os.path.join(prefix, 'params.tar.gz')} --files-from -"
        )
        os.system(f'rm -f ./modelsaved/*.pth')
        print('all operations complete')


if __name__ == '__main__':
    '''
    log = LogDocument(args)
    log.write()
    '''
    pass
