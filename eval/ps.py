import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# lst_colors = [
#     '#f9dbbd',
#     '#ffa5ab',
#     '#da627d',
#     '#a53860',
#     '#450920',
# ]
lst_colors = [
    # '#f9dbbd',
    "#ffe5d9",
    "#ffcfd2",
    '#ffa5ab',
    '#da627d',
    '#a53860',
    # '#450920',
]
from matplotlib.colors import LinearSegmentedColormap
custom1 = LinearSegmentedColormap.from_list(
    name='pink',
    colors=lst_colors,
)

dataset = 'domainnet'

if dataset == 'cifar10':
    lst_methods = {
        'ER': '/data/output-ai/fahad.sarfraz/SYNERgy/er_cifar10/results/class-il/seq-cifar10/er/er-cifar10-%s-param-v1-r3/',
        'DER++': '/data/output-ai/fahad.sarfraz/SYNERgy/derpp_cifar10/results/class-il/seq-cifar10/derpp/derpp-cifar10-%s-param-v1-r3/',
        'cls-er': '/data/output-ai/fahad.sarfraz/SYNERgy/analysis_models/cifar10/cil_er/results/class-il/seq-cifar10/dualmeanerv6/er-c10-%s-param-v1-s-0_stable_ema_model/',
        'aux': '/data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1/results/class-il/seq-cifar10/derpp_mm_eman1_model/derppema-n1m-%s-seq-cifar10-kl1.0-0.1-up0.2-s2_ema_net1/',
    }
    num_tasks = 5
    annot = True
elif dataset == 'cifar100':
    lst_methods = {
        'DER++': '/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/seq-cifar100/derpp/cll-derpp-%s-seq-cifar100-s0/',
        'ER': '/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/seq-cifar100/er/cll-er-%s-seq-cifar100-s0/',
        'cls-er': '/data/output-ai/fahad.sarfraz/lll_baselines/results/class-il/seq-cifar100/clser/c100-5-%s-param-v4-0.05-0.1s-0_stable_model/',
        'aux': '/data/output-ai/shruthi.gowda/continual/cifar100/results/class-il/seq-cifar100/derpp_mm_eman1/cll-cif100%s-a0.1b0.5-lr0.03-l20.10.01-up0.06-g4-s0_ema_net1/',
    }
    num_tasks = 5
    annot = True
elif dataset == 'domainnet':
    lst_methods = {
        'DER++': '/data/output-ai/shruthi.gowda/continual/domainNet/base/base_tiny_params/tiny_param/results/domain-il/domain-net/derpp/cll-derpp-%s-domain-netv2-s0',
        'ER': '/data/output-ai/shruthi.gowda/continual/domainNet/base/base_tiny_params/tiny_param/results/domain-il/domain-net/er/cll-er-%s-domain-netv2-s0',
        'cls-er': '/data/output-ai/shruthi.gowda/continual/domainNet/method/results/domain-il/domain-net/clser/cll-cls-%s-domain-netv2-64-l20.1-0.01-s0_stable_model',
        'aux': '/data/output-ai/shruthi.gowda/continual/domainNet/method/results/domain-il/domain-net/derpp_mm_eman1/cll-%s-domain-netv2-lr0.03-l20.10.01-em0.06-s2_ema_net1',
    }
    num_tasks = 6
    annot = True

x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After T{i}" for i in range(1, num_tasks + 1)]

n_rows, n_cols = 1, 1
fig, ax = plt.subplots(n_rows, n_cols, figsize=(13, 11)) # sharey=True, sharex=True)


annot = True

fmt = '.1f'
# fmt = '%d'
font = 18

lst_method = ['ER', 'DER++', 'cls-er', 'aux']
buffer_size = 200

k = 0
x =  np.arange(4)
pl = []
st = []
tr = []
for n, method in enumerate(lst_method):
    perf_path = os.path.join(lst_methods[method] % buffer_size, 'task_performance.txt')

    np_perf = np.loadtxt(perf_path)

    # p = np_perf[0][0] + np_perf[1][1] + np_perf[2][2] + np_perf[3][3] + np_perf[4][4] + np_perf[5][5]
    # pl.append(p/6)
    # s = np_perf[5][0] + np_perf[5][1] + np_perf[5][2] + np_perf[5][3] + np_perf[5][4] #np_perf[4][0] + np_perf[4][1] + np_perf[4][2] + np_perf[4][3]
    # st.append(s/6)

    p = np_perf[0][0] + np_perf[1][1] + np_perf[2][2] + np_perf[3][3] + np_perf[4][4]
    pl.append(p/4)
    s = np_perf[4][0] + np_perf[4][1] + np_perf[4][2] + np_perf[4][3]
    st.append(s/4)
    t =(1/p)+(1/s)
    tr.append(1/t)

width = 0.4
ax.bar(x - width/2, pl, width, color='#ffc2d1', align='center', label='Plasticity', hatch='/', alpha=.99)
ax.bar(x + width/2, st, width, color='#d88c9a', align='center', label='Stability', hatch='\\', alpha=.99)
# ax.bar(x + 0.2, tr, width=0.2, color='#a53860', align='center', label='Trade-off')
import matplotlib as mpl
mpl.rcParams["hatch.color"] = 'red'

font = 34

y_ticks = ax.yaxis.get_major_ticks()

plt.ylim([0, 75])
# plt.yticks(np.arange(0, 90, 10))
# plt.yticks[2].set_visible(False)

# y_ticks[-1].label1.set_visible(False)

plt.xticks(x, ['ER', 'DER++', 'CLS-ER', 'DUCA'])
plt.xticks(fontsize=35)
plt.ylabel("Accuracy", fontsize=font)
plt.legend(fontsize=36, loc="upper center", ncol=2, frameon=False, )
plt.tick_params(axis="y", labelsize=font)      # To change the y-axis
plt.show()

fig.savefig(f'/volumes2/continual_learning/paper/icml/ps_dom_500_avg.png', bbox_inches='tight')
fig.savefig(f'/volumes2/continual_learning/paper/icml/ps_dom_500_avg.pdf', bbox_inches='tight')
