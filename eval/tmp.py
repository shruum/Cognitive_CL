import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch


dataset = 'cifar10'
lst_methods = {
    'der': '/data/output-ai/fahad.sarfraz/SYNERgy/derpp_cifar10/task_models/seq-cifar10/derpp-cifar10-%s-param-v1/task_%s_model.ph',
    'er': '/data/output-ai/fahad.sarfraz/SYNERgy/er_cifar10/task_models/seq-cifar10/er-cifar10-%s-param-v1-r3/task_%s_model.ph',
    'cls-er': '/data/output-ai/fahad.sarfraz/SYNERgy/analysis_models/cifar10/cil_er/task_models/seq-cifar10/er-c10-%s-param-v1-s-0/task_%s_stable_model.ph',
    'synergy': '/data/output-ai/fahad.sarfraz/SYNERgy/syncer_cifar10/checkpoints/seq-cifar10/synergy-cifar10-%s-param-v1/ema_model_task%s.pt'
}

method = 'synergy'
buffer_size = 500
num_tasks = 5

lst_models = ['der', 'er', 'cls-er', 'synergy']


def get_normalized_params(model):
    lst_params = []
    for param in model.parameters():
        lst_params.append(param.view(-1) / param.max())

    return torch.cat(lst_params)




# lst_diff = []
# for method in lst_models:
#
#     lst_params = []
#     for task_id in range(num_tasks):
#         if method == 'synergy':
#             model_path = lst_methods[method] % (buffer_size, task_id)
#         else:
#             model_path = lst_methods[method] % (buffer_size, task_id + 1)
#         model = torch.load(model_path)
#         # lst_params.append(model.get_params() / model.get_params().max())
#         lst_params.append(get_normalized_params(model))
#
#     sim_mat = np.zeros((num_tasks, num_tasks))
#     sim = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
#
#     for i in range(num_tasks):
#         for j in range(num_tasks):
#             if i >= j:
#                 cos_sim = sim(lst_params[i], lst_params[j])
#
#                 # diff1 = ((lst_params[i] - lst_params[j])**2).mean()
#                 diff2 = (abs(lst_params[i] - lst_params[j])).mean()
#
#                 sim_mat[i][j] = diff2
#
#     lst_diff.append(sim_mat)

# ======================================================================================================================
# Layer wise similarity
# ======================================================================================================================
lst_diff = []
for method in lst_models:

    lst_params = []
    for task_id in range(num_tasks):
        if method == 'synergy':
            model_path = lst_methods[method] % (buffer_size, task_id)
        else:
            model_path = lst_methods[method] % (buffer_size, task_id + 1)

        model = torch.load(model_path)
        # lst_params.append(model.get_params() / model.get_params().max())
        lst_params.append(list(model.parameters()))

    sim_mat = np.zeros((num_tasks, num_tasks))
    sim = torch.nn.CosineSimilarity(dim=0, eps=1e-08)

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i >= j:
                model_diff = 0
                count = 0
                for param1, param2 in zip(lst_params[i], lst_params[j]):
                    # param1, param2 = param1.view(-1), param2.view(-1)
                    param1, param2 = param1.view(-1) / param1.max(), param2.view(-1) / param2.max()
                    model_diff += sim(param1, param2)
                    # model_diff += ((param1 - param2)**2).sum()
                    # model_diff += (abs(param1 - param2)).sum()
                    count += 1

                model_diff /= count
                sim_mat[i][j] = model_diff

    lst_diff.append(sim_mat)

v_max = 0
v_min = 1000

for mat in lst_diff:
    mat[mat == 0] = mat.max()
    max, min = mat.max(), mat.min()

    if v_max < max:
        v_max = max
    if v_min > min:
        v_min = min


import seaborn as sns
# fig, ax = plt.subplots()

n_rows, n_cols = 1, 4
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 4), sharey=True, sharex=True)

annot = True
fmt = '.2f'
# fmt = '%d'
fontsize = 10

# plt.rcParams['text.usetex'] = True

x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
# x_labels = [r"$\theta_{T%s}$" % i for i in range(1, num_tasks + 1)]
# y_labels = [r"$\theta_{T%s}$" % i for i in range(1, num_tasks + 1)]

for n, mat in enumerate(lst_diff):

    matrix = np.triu(np.ones_like(mat)) - np.identity(mat.shape[0])

    if n == 3:
        cbar_ax = fig.add_axes([0.91, 0.18, 0.02, 0.6])
        im = sns.heatmap(mat, ax=ax[n], vmax=v_max, vmin=v_min, mask=matrix, annot=annot, cmap='YlOrBr', alpha=0.85, cbar=True, fmt=fmt, linewidths=.5, annot_kws={"size": fontsize}, cbar_ax=cbar_ax)
    else:
        im = sns.heatmap(mat, ax=ax[n], vmax=v_max, vmin=v_min, mask=matrix, annot=annot, cmap='YlOrBr', alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": fontsize})

    ax[n].set_xticks(np.arange(len(x_labels)) + 0.5)
    ax[n].set_yticks(np.arange(len(y_labels)) + 0.5)
    ax[n].set_xticklabels(x_labels, ha='center', fontsize=9)
    ax[n].set_yticklabels(y_labels, rotation=0, va='center', fontsize=9)
    ax[n].set_aspect('equal', adjustable='box')

    ax[n].axhline(y=0, color='k', linewidth=1)
    ax[n].axhline(y=mat.shape[1], color='k', linewidth=2)
    ax[n].axvline(x=0, color='k', linewidth=1)
    ax[n].axvline(x=mat.shape[1], color='k', linewidth=2)


ax[0].set_title('ER', fontsize=12)
ax[1].set_title('DER++', fontsize=12)
ax[2].set_title('CLS-ER', fontsize=12)
ax[3].set_title('SYNERgy', fontsize=12)

# ax = sns.heatmap(sim_mat, annot=True)
plt.show()
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# fig.savefig(f'analysis/figures/weight_diff/{dataset}_{buffer_size}.png', bbox_inches='tight')
# fig.savefig(f'analysis/figures/weight_diff/{dataset}_{buffer_size}.pdf', bbox_inches='tight')