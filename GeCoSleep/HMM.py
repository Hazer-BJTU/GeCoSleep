import torch
import numpy as np
from hmmlearn import hmm


def balance_seq(seqs, number, eps=1e-5, num_classes=5):
    class_counts = torch.bincount(seqs.view(-1), minlength=num_classes).float()
    class_weights = 1.0 / (class_counts + eps)
    class_weights = class_weights.pow(0.5)
    class_weights = class_weights / class_weights.sum()
    seq_weights = torch.mean(class_weights[seqs], dim=1)
    seq_weights = seq_weights / seq_weights.sum()
    idx = torch.multinomial(seq_weights, number, replacement=True)
    return seqs[idx]


class HMMSequenceGenerator:
    def __init__(self, num_components, iterations, device, random_state=42):
        self.num_components = num_components
        self.iterations = iterations
        self.device = device
        self.random_state = random_state
        self.model = hmm.CategoricalHMM(n_components=num_components, n_iter=iterations, random_state=random_state)
        self.fitted = False
        self.length = None

    def fit(self, seqs):
        num_samples = seqs.shape[0]
        '''seqs = balance_seq(seqs, num_samples)'''
        num_samples, self.length = seqs.shape[0], seqs.shape[1]
        seqs = seqs.cpu().numpy().tolist()
        self.model.fit(seqs, lengths=num_samples)
        self.fitted = True

    def generate_one(self):
        if not self.fitted:
            raise RuntimeError('HMM model not fitted!')
        gen_seq, _ = self.model.sample(self.length)
        output = torch.tensor(gen_seq.flatten().tolist(), dtype=torch.int64, device=self.device, requires_grad=False)
        return output


class HMMTaskGenerator:
    def __init__(self, num_tasks, device):
        self.num_tasks = num_tasks
        self.device = device
        self.models = [HMMSequenceGenerator(10, 50, device) for _ in range(num_tasks)]
        self.task_samples = []
        self.total_samples = 0
        self.cnt = 0

    def clear_samples(self):
        self.task_samples = []
        self.total_samples = 0

    def add_sample(self, y):
        self.task_samples.append(y)
        self.total_samples += y.shape[0]

    def ready(self):
        samples = torch.cat(self.task_samples, dim=0)
        self.models[self.cnt].fit(samples)
        self.cnt += 1

    def generate(self, t):
        outputs = []
        for i in range(t.shape[0]):
            seqs = self.models[t[i]].generate_one()
            outputs.append(seqs.unsqueeze(dim=0))
        return torch.cat(outputs, dim=0)


if __name__ == '__main__':
    pass
