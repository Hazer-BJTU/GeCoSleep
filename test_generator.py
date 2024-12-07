import torch
import random
from torch import nn
from models import *
from generator import *
from matplotlib import pyplot as plt
from data_preprocessing import *


if __name__ == '__main__':
    device = torch.device(f'cuda:{0}')
    generator_path = './modelsaved/generator_task0_fold0.pth'
    critic_path = './modelsaved/generative_task0_fold0.pth'
    critic = SleepNet(2, 0.5)
    critic.load_state_dict(torch.load(critic_path, map_location=device, weights_only=True))
    critic.eval()
    generator = EEGVAE(2)
    generator.load_state_dict(torch.load(generator_path, map_location=device, weights_only=True))
    zs = [torch.randn((1, 8, 3750)), torch.randn(1, 16, 937), torch.randn(1, 32, 234), torch.randn(1, 128, 58)]
    X_hat = generator.decoder(zs)
    labels = critic(X_hat)
    print(torch.argmax(labels, dim=1))
    X_hat = X_hat.detach().cpu().numpy()
    series = np.arange(3000)
    datas, labels = load_data_isruc1('/home/ShareData/ISRUC-1/ISRUC-1', 10, ['C4_A1', 'LOC_A2'], 3)
    i = random.randint(0, len(datas) - 1)
    j = random.randint(0, len(datas[i]) - 1)
    ground_truth = datas[i][j]
    labels_truth = critic(ground_truth.unsqueeze(0))
    print(torch.argmax(labels_truth, dim=1))
    figure = plt.figure(figsize=(18, 6))
    figure.canvas.manager.set_window_title('Results')
    figure.add_subplot(4, 1, 1)
    plt.title('generated-large-scale')
    plt.plot(series, X_hat[0, 4, 0, :], c='r')
    figure.add_subplot(4, 1, 2)
    plt.title('real-large-scale')
    plt.plot(series, ground_truth[4, 0, :], c='b')
    series = np.arange(300)
    figure.add_subplot(4, 1, 3)
    plt.title('generated-small-scale')
    plt.plot(series, X_hat[0, 4, 0, 1500:1800], c='r')
    figure.add_subplot(4, 1, 4)
    plt.title('real-small-scale')
    plt.plot(series, ground_truth[4, 0, 1500:1800], c='b')
    plt.tight_layout()
    plt.savefig('generate_result.png')
