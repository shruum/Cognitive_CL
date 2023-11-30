import yaml
import os

# job = yaml.load(open(r'template.yaml'))
# job = yaml.load(open(r'scripts/template_gpu4.yaml'))

best_params_domain = {
    200: {'lr': 0.05,
          'minibatch_size': 32,
          'alpha': 0.1,
          'beta': 1.0,
          'batch_size': 32,
          'n_epochs': 50,
          'aux': 'shape',
          'img_size': 64,
          'shape_filter': 'sobel',
          'shape_upsample_size': 128,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': 'True',
          'loss_types': ['kl', 'l2'],
          'loss_wt_lst1': [0.1, 0.01],
          'loss_wt_lst2': [0.1, 0.01],
          'ema_alpha_lst': [0.999],
          'ema_update_freq_lst': [0.06, 0.08]
          },
    500: {'lr': 0.05,
          'minibatch_size': 32,
          'alpha': 0.2,
          'beta': 0.5,
          'batch_size': 32,
          'n_epochs': 50,
          'aux': 'shape',
          'img_size': 64,
          'shape_filter': 'sobel',
          'shape_upsample_size': 128,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': 'True',
          'loss_types': ['kl', 'l2'],
          'loss_wt_lst1': [0.1, 0.01],
          'loss_wt_lst2': [0.1, 0.01],
          'ema_alpha_lst': [0.999],
          'ema_update_freq_lst':[0.06, 0.08]
          },
}

best_params_rot = {
    200: {
        'idt': 'v1',
        'lr': 0.1,
        'minibatch_size': 128,
        'alpha': 0.75,
        'beta': 0.5,
        'batch_size': 128,
        'n_epochs': 1,
        'aux': 'shape',
        'img_size': 28,
        'shape_filter': 'sobel',
        'shape_upsample_size': 56,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
        'loss_type': ['kl'],
        'loss_wt_lst1': [0.1, 0.01],
        'loss_wt_lst2' : [0.1, 0.01],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [1.0],
    },
    500: {
        'idt': 'v1',
        'lr': 0.2,
        'minibatch_size': 128,
        'alpha': 0.5,
        'beta': 1.0,
        'batch_size': 128,
        'n_epochs': 1,
        'aux': 'shape',
        'img_size': 28,
        'shape_filter': 'sobel',
        'shape_upsample_size': 56,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
        'loss_type': ['kl'],
        'loss_wt_lst1': [0.1, 0.01],
        'loss_wt_lst2' : [0.1, 0.01],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [1.0],
    },
}

lst_buffer_size = [200] #, 500]
num_runs = 1
start_seed = 0
count = 0
loss_types = ['l2'] #, 'l2']
dir_aux_lst = {True, False}
dir_aux = True

lst_datasets = [
    ('domain-net', best_params_domain),
    #('rot-mnist', best_params_rot),
]

for seed in range(start_seed, start_seed + num_runs):
    for dataset, params in lst_datasets:
        for buffer_size in lst_buffer_size:
            train_params = params[buffer_size]
            for ema_update_freq in train_params['ema_update_freq_lst']:
                for ema_alpha in train_params['ema_alpha_lst']:
                    for loss_type in loss_types:
                        for loss_wt1 in train_params['loss_wt_lst1']:
                            for loss_wt2 in train_params['loss_wt_lst2']:
                                exp_id = f"derppema-n1m-{buffer_size}-{dataset}v2-64-{loss_type}{loss_wt1}-{loss_wt2}-s{seed}"
                                job_args = f"/data/users/shruthi.gowda/workspace/envs/continual/bin/python /data/users/shruthi.gowda/workspace/continual/mammothssl/main.py  \
                                    --experiment_id {exp_id} \
                                    --seed {seed} \
                                    --model derpp_mm_eman1 \
                                    --dataset {dataset} \
                                    --buffer_size {buffer_size} \
                                    --aux {train_params['aux']} \
                                    --alpha_mm {train_params['alpha']} {train_params['alpha']} \
                                    --beta_mm {train_params['beta']} {train_params['beta']} \
                                    --lr {train_params['lr']} \
                                    --minibatch_size {train_params['minibatch_size']} \
                                    --n_epochs {train_params['n_epochs']} \
                                    --img_size {train_params['img_size']} \
                                    --shape_filter {train_params['shape_filter']} \
                                    --shape_upsample_size {train_params['shape_upsample_size']} \
                                    --sobel_gauss_ksize {train_params['sobel_gauss_ksize']} \
                                    --sobel_ksize {train_params['sobel_ksize']} \
                                    --sobel_upsample {train_params['sobel_upsample']} \
                                    --batch_size {train_params['batch_size']} \
                                    --output_dir /data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1_domain_desk \
                                    --loss_type {loss_type} \
                                    --loss_wt {loss_wt1} {loss_wt2} \
                                    --ema_alpha {ema_alpha} \
                                    --dir_aux \
                                    --ema_update_freq {ema_update_freq} \
                                    --tensorboard \
                                    --csv_log \
                                    "

                                count += 1
                                os.system(job_args)

print('%s jobs counted' % count)

