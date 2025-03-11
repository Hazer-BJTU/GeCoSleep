import math
import torch
import torch.nn as nn
import random
from clnetworks import CLnetwork
from metric import ConfusionMatrix, evaluate_tasks

def get_flat_grad(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads) if len(grads) > 0 else None

def set_flat_grad(model, flat_grad):
    pointer = 0
    for param in model.parameters():
        if param.grad is not None:
            num_param = param.numel()
            param.grad.copy_(flat_grad[pointer:pointer + num_param].view(param.size()))
            pointer += num_param

class TA_GEMnetwork(CLnetwork):
    def __init__(self, args, fold_num, logs):
        super(TA_GEMnetwork, self).__init__(args, fold_num, logs)
        self.memory_clusters = []
        self.num_clusters = self.args.num_clusters
        self.cluster_size = self.args.batch_size
        self.mem_batch = self.args.batch_size

    def start_task(self):
        super(TA_GEMnetwork, self).start_task()

    def start_epoch(self):
        super(TA_GEMnetwork, self).start_epoch()
    
    def observe(self, X, y, first_time=False):
        X, y = X.to(self.device), y.to(self.device)
        self.update_memory(X, y)
        self.optimizer.zero_grad()
        y_hat = self.net(X)
        L_current = torch.mean(self.loss(y_hat, y.view(-1)))
        L_current.backward()
        g_cur = get_flat_grad(self.net)
        if self.memory_available():
            mem_X, mem_y = self.sample_memory(self.mem_batch)
            mem_X, mem_y = mem_X.to(self.device), mem_y.to(self.device)
            self.optimizer.zero_grad()
            y_mem_hat = self.net(mem_X)
            L_mem = torch.mean(self.loss(y_mem_hat, mem_y.view(-1)))
            L_mem.backward()
            g_ref = get_flat_grad(self.net)
            prod = torch.dot(g_cur, g_ref)
            if prod < 0:
                proj_grad = g_cur - (prod / (g_ref.norm()**2 + 1e-12)) * g_ref
            else:
                proj_grad = g_cur
            set_flat_grad(self.net, proj_grad)
        self.optimizer.step()
        self.train_loss += L_current.item()
        self.cnt += 1
        self.confusion_matrix.count_task_separated(y_hat, y, 0)

    def end_epoch(self, valid_dataset):
        super(TA_GEMnetwork, self).end_epoch(valid_dataset)

    def end_task(self, dataset=None):
        self.task += 1
        self.best_net_memory.append(self.best_net)

    def memory_available(self):
        for cluster in self.memory_clusters:
            if len(cluster['samples']) > 0:
                return True
        return False

    def sample_memory(self, batch_size):
        all_samples = []
        all_labels = []
        for cluster in self.memory_clusters:
            for (X_sample, y_sample, _) in cluster['samples']:
                all_samples.append(X_sample.unsqueeze(0))
                all_labels.append(y_sample.unsqueeze(0))
        if len(all_samples) == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        indices = random.choices(range(len(all_samples)), k=batch_size)
        mem_X = torch.cat([all_samples[i] for i in indices], dim=0)
        mem_y = torch.cat([all_labels[i] for i in indices], dim=0)
        return mem_X, mem_y

    def update_memory(self, X, y):
        with torch.no_grad():
            feats = self.net.features(X)
            feats = feats.mean(dim=1).cpu()
        batch_size = X.shape[0]
        for i in range(batch_size):
            sample_X = X[i].detach().cpu()
            sample_y = y[i].detach().cpu()
            sample_feat = feats[i]
            self._update_memory_sample(sample_X, sample_y, sample_feat)

    def _update_memory_sample(self, sample_X, sample_y, sample_feat):
        if len(self.memory_clusters) < self.num_clusters:
            new_cluster = {'center': sample_feat.clone(), 'samples': [(sample_X, sample_y, sample_feat.clone())]}
            self.memory_clusters.append(new_cluster)
            return
        min_dist = None
        assigned_cluster = None
        for cluster in self.memory_clusters:
            dist = torch.norm(sample_feat - cluster['center'])
            if min_dist is None or dist < min_dist:
                min_dist = dist
                assigned_cluster = cluster
        if len(assigned_cluster['samples']) < self.cluster_size:
            assigned_cluster['samples'].append((sample_X, sample_y, sample_feat.clone()))
        else:
            assigned_cluster['samples'].pop(0)
            assigned_cluster['samples'].append((sample_X, sample_y, sample_feat.clone()))
        feats = [s[2] for s in assigned_cluster['samples']]
        assigned_cluster['center'] = torch.stack(feats, dim=0).mean(dim=0)

