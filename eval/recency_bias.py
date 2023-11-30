from __future__ import print_function

import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../"))
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def get_task_probabilities(model, device, data_loader, task_dist):
    model.eval()
    lst_logits = []

    with torch.no_grad():
        for input, label in data_loader:
            input, label = input.to(device), label.to(device)
            logits = model(input)

            lst_logits.append(logits.detach().cpu())

    logits = torch.cat(lst_logits).to(device)
    softmax_scores = F.softmax(logits, dim=1)

    lst_prob = []
    for task in task_dist:
        prob = torch.mean(softmax_scores[:, task_dist[task][0]: task_dist[task][1]])
        lst_prob.append(prob.item())

    np_prob = np.array(lst_prob)
    np_prob = np_prob / np.sum(np_prob)

    return np_prob


def get_task_probabilities_ensemble(model1, model2, device, data_loader, task_dist):

    model1.eval()
    model2.eval()

    lst_logits = []

    with torch.no_grad():
        for input, label in data_loader:
            input, label = input.to(device), label.to(device)
            logits1 = model1(input)
            logits2 = model2(input)

            logits = logits1 + logits2

            lst_logits.append(logits.detach().cpu())

    logits = torch.cat(lst_logits).to(device)
    softmax_scores = F.softmax(logits, dim=1)

    lst_prob = []
    for task in task_dist:
        prob = torch.mean(softmax_scores[:, task_dist[task][0]: task_dist[task][1]])
        lst_prob.append(prob.item())

    np_prob = np.array(lst_prob)
    np_prob = np_prob / np.sum(np_prob)

    return np_prob


# =============================================================================
# Load Dataset
# =============================================================================
# CIFAR 10
TRANSFORM = transforms.Compose(
    [
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2615))
     ]
)

# CIFAR 10
TRANSFORM_AUX = transforms.Compose(
    [
         transforms.ToTensor(),
     ]
)

# =============================================================================
# Evaluate Calibration Plots
# =============================================================================
lst_buffer_size = [200] #, 500, 5120]
# lst_buffer_size = [200]
# lst_buffer_size = [5120]
device = 'cuda'


dataset = CIFAR10('data', train=False, download=True, transform=TRANSFORM)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
dataset_aux = CIFAR10('data', train=False, download=True, transform=TRANSFORM_AUX)
data_loader_aux = torch.utils.data.DataLoader(dataset_aux, batch_size=32, shuffle=False, num_workers=4)
task_dist = {
    'task1': (0, 2),
    'task2': (2, 4),
    'task3': (4, 6),
    'task4': (6, 8),
    'task5': (8, 10),
}
lst_methods = {
    'der': '/data/output-ai/fahad.sarfraz/SYNERgy/derpp_cifar10/task_models/seq-cifar10/derpp-cifar10-200-param-v1/task_5_model.ph',
    'er': '/data/output-ai/fahad.sarfraz/SYNERgy/analysis_models/cifar10/er/task_models/seq-cifar10/er-c10-b-200-r-2/task_5_model.ph',
    'cls-er': '/data/output-ai/fahad.sarfraz/SYNERgy/analysis_models/cifar10/cil_er/task_models/seq-cifar10/er-c10-200-param-v1-s-0/task_5_stable_model.ph',
    # 'aux': '/data/output-ai/shruthi.gowda/continual/cifar10/results/class-il/seq-cifar10/derpp_mm_eman1/cll-cif100200-a0.1b0.5-lr0.03-l20.10.1-up0.2-urgent-s0_ema_net1/ema_net1_final.pth',
    'aux': '/data/output-ai/shruthi.gowda/continual/method/cifar10/results/class-il/seq-cifar10/derpp_mm_eman1/cll-cif100200-a0.1b0.5-lr0.03-l20.10.1-up0.1-urgent-s1_ema_net1/ema_net1_final.pth'
}

# dataset = CIFAR100('data', train=False, download=True, transform=TRANSFORM)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
# dataset_aux = CIFAR100('data', train=False, download=True, transform=TRANSFORM_AUX)
# data_loader_aux = torch.utils.data.DataLoader(dataset_aux, batch_size=32, shuffle=False, num_workers=4)
# task_dist = {
#     'task1': (0, 20),
#     'task2': (20, 40),
#     'task3': (40, 60),
#     'task4': (60, 80),
#     'task5': (80, 100),
# }
# lst_methods = {
#     'der': '/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/seq-cifar100/derpp/cll-derpp-200-seq-cifar100-s0/net_final.pth',
#     'er': '/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/seq-cifar100/er/cll-er-200-seq-cifar100-s0/net_final.pth',
#     'cls-er': '/data/output-ai/fahad.sarfraz/lll_baselines/task_models/seq-cifar100/c100-5-200-param-v4-0.05-0.1s-0/stable_model.ph',
#     'aux': '/data/output-ai/shruthi.gowda/continual/cifar100/results/class-il/seq-cifar100/derpp_mm_eman1/cll-cif100200-a0.15b0.5-lr0.03-l20.10.01-up0.06-urgent-s0_ema_net1/ema_net1_final.pth'
# }

lst_tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5']

lst_colors = [
    '#FACCFF',
    '#DCB0FF',
    '#BE93FD',
    '#A178DF',
    '#845EC2',
]

lst_colors = [
    # '#f9dbbd',
    "#ffe5d9",
    "#ffcfd2",
    '#ffa5ab',
    '#da627d',
    '#a53860',
    # '#450920',
]


for buffer_size in lst_buffer_size:
    # buffer_size = 500
    print('=' * 50)
    print(f'Buffer Size = {buffer_size}')
    print('=' * 50)

    ind = np.arange(len(lst_tasks))
    width = 0.1

    results = {}
    model_path = lst_methods['er']
    model = torch.load(model_path).to(device)
    er_prob = get_task_probabilities(model, 'cuda', data_loader, task_dist)
    # ax.bar(ind, task_prob, width, label='ER', color='firebrick')

    model_path =lst_methods['der']
    model = torch.load(model_path).to(device)
    der_prob= get_task_probabilities(model, 'cuda', data_loader, task_dist)
    # ax.bar(ind + width, task_prob, width, label='DER++', color='steelblue')

    model_path = lst_methods['cls-er']
    model = torch.load(model_path).to(device)
    cls_prob = get_task_probabilities(model, 'cuda', data_loader, task_dist)
    # ax.bar(ind + 2 * width, task_prob, width, label='CLS-ER-Stable', color='turquoise')

    model_path = lst_methods['aux']
    model = torch.load(model_path).to(device)
    aux_prob = get_task_probabilities(model, 'cuda', data_loader_aux, task_dist)

    # plastic_model_path = os.path.join(models_repo, lst_methods['cls-er'] % buffer_size, f'task_5_plastic_model.ph')
    # plastic_model = torch.load(plastic_model_path).to(device)
    # results['CLS-ER-Plastic'] = get_task_probabilities(plastic_model, 'cuda', data_loader, task_dist)
    # ax.bar(ind + 3 * width, task_prob, width, label='CLS-ER-Plastic', color='rebeccapurple')

    # model_path = os.path.join(models_repo, lst_methods['cls-er'] % buffer_size, f'task_5_model.ph')
    # model = torch.load(model_path).to(device)
    # results['CLS-ER-Model'] = get_task_probabilities(model, 'cuda', data_loader, task_dist)
    # ax.bar(ind + 4 *width, task_prob, width, label='CLS-ER-Model', color='mediumpurple')

    # task_prob = get_task_probabilities_ensemble(plastic_model, stable_model, 'cuda', data_loader, task_dist)
    # ax.bar(ind + 5 * width, task_prob, width, label='CLS-ER-Ensemble', color='green')
    prob = np.vstack((er_prob, der_prob, cls_prob, aux_prob))
    n_methods, n_tasks = prob.shape
    print(n_tasks)

    fig, ax = plt.subplots(figsize=(9, 7))
    barWidth = 0.14
    avg = []
    for i in range(n_tasks):
        x = np.arange(n_methods) + i * barWidth
        plt.bar(x, prob[:, i], color=lst_colors[i], width=barWidth, label=f'Task {i + 1}')

    for i in range(n_methods):
        avg.append(np.mean(abs([0.2,0.2,0.2, 0.2, 0.2] - prob[i,:])))

    font = 24
    plt.ylabel('Task Probability', fontsize=font)
    plt.xticks([r + 2 * barWidth for r in range(n_methods)], ['ER', 'DER++', 'CLS-ER', 'DUCA'], fontsize=font)
    plt.yticks(fontsize=font)
    plt.legend(fontsize=19, )
    plt.xlabel('Method', fontsize=font)

    # plt.axhline(y=0.2, color='r', linestyle='--')

    # plt.ylim(0, 0.85)
    plt.show()
    print(avg)
    fig.savefig(r'/volumes2/continual_learning/paper/icml/rbias_cif10_200_1.png', bbox_inches='tight')
    fig.savefig(r'/volumes2/continual_learning/paper/icml/rbias_cif10_200.pdf', dpi=600, bbox_inches='tight')