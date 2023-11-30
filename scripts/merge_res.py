import os
import glob
import pandas as pd

# specifying the path to csv files
# path = "/data/output-ai/shruthi.gowda/continual/der_base/results/task-il"
path = "/data/output-ai/shruthi.gowda/continual/clmm_derpp_v3/results/class-il/seq-cifar10"

extension = 'csv'
all_filenames = [i for i in glob.glob(path + '/*/*s0_ema_net1/*.{}'.format(extension))]
#all_filenames = [i for i in glob.glob(path + '/*/*/*s0_ema_net2/*.{}'.format(extension))]
combined_csv = []

#combine all files in the list
# for f in all_filenames:
#     a = pd.read_csv(f, sep=',')
#     a.drop('loss_type', inplace=True, axis=1)
#     a.drop('loss_wt_kl', inplace=True, axis=1)
#     a.drop('loss_wt_at', inplace=True, axis=1)
#     a.drop('loss_wt_l2', inplace=True, axis=1)
#     combined_csv.append(a)

combined_csv = [pd.read_csv(f, sep=',') for f in all_filenames ]
combined_csv = pd.concat(combined_csv, ignore_index=True)
#export to csv
combined_csv.to_csv(os.path.join(path,"clsil_emanet1.csv"), index=False, encoding='utf-8-sig')
