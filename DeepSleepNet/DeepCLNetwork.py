from .DeepSleepNet import DeepSleepNet
from clnetworks import CLnetwork
from models import init_weight


class DeepCLNetwork(CLnetwork):
    def __init__(self, args, fold_num, logs):
        super(DeepCLNetwork, self).__init__(args, fold_num, logs)
        self.net = DeepSleepNet(self.num_channels, dropout=args.dropout)
        self.net.apply(init_weight)


if __name__ == '__main__':
    pass
