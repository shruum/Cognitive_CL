import yaml
import os


best_params_seqcif10 = {
    200: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha': 0.3,
        'beta': 0.5,
        'batch_size': 32,
        'n_epochs': 50,
        'aux': 'shape',
        'img_size': 32,
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
        'loss_type': ['kl'],
        'loss_wt_lst1': [0.1, 0.01],
        'loss_wt_lst2': [0.1, 0.01],
        'loss_wt_lst3': [0.1, 0.01],
        'loss_wt_lst4': [0.1, 0.01],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [0.09, 0.3]
    },
    500: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha': 0.2,
        'beta': 0.5,
        'batch_size': 32,
        'n_epochs': 50,
        'aux': 'shape',
        'img_size': 32,
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
        'loss_type': ['kl'],
        'loss_wt_lst1': [0.1, 0.01],
        'loss_wt_lst2': [0.1, 0.01],
        'loss_wt_lst3': [0.1, 0.01],
        'loss_wt_lst4': [0.1, 0.01],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [0.09, 0.3]
    },
}

best_params_seqcif100 = {
    200: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha': 0.1,
        'beta': 0.5,
        'batch_size': 32,
        'n_epochs': 20,
        'aux': 'shape',
        'img_size': 32,
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
        'loss_type': ['kl'],
        'loss_wt_lst1': [0.1,],
        'loss_wt_lst2': [0.01],
        'loss_wt_lst3': [0.1],
        'loss_wt_lst4': [0.01],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [0.1]
    },
    500: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha': 0.2,
        'beta': 0.5,
        'batch_size': 32,
        'n_epochs': 50,
        'aux': 'shape',
        'img_size': 32,
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
        'loss_type': ['kl'],
        'loss_wt_lst1': [0.1],
        'loss_wt_lst2': [0.01],
        'loss_wt_lst3': [0.1],
        'loss_wt_lst4': [0.01],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [0.06]
    },
}

best_params_seqtiny = {
    200: {'lr': 0.03,
          'minibatch_size': 32,
          'alpha': [0.1],
          'beta': [1.0, 0.5],
          'batch_size': 32,
          'n_epochs': 50,
          'aux': 'shape',
          'img_size': 64,
          'shape_filter': 'sobel',
          'shape_upsample_size': 128,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': ['True'],
          'loss_types': ['l2'],
          'loss_wt_lst1': [0.01],
          'loss_wt_lst2': [0.1],
          'loss_wt_lst3': [0.03, 0.04],
          'loss_wt_lst4': [0.01, 0.02],
          'ema_update_freq_lst': [0.04, 0.05]
          },
    500: {'lr': 0.03,
          'minibatch_size': 32,
          'alpha': [0.1, 0.2],
          'beta': [0.5,1.0],
          'batch_size': 32,
          'n_epochs': 50,
          'aux': 'shape',
          'img_size': 64,
          'shape_filter': 'sobel',
          'shape_upsample_size': 128,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': ['True'],
          'loss_types': ['l2'],
          'loss_wt_lst1': [0.01],
          'loss_wt_lst2': [0.01, 0.02],
          'loss_wt_lst3': [0.06, 0.1],
          'loss_wt_lst4': [0.01, 0.02],
          'ema_update_freq_lst': [0.05, 0.06]
          },
}


lst_buffer_size = [200, 500]
num_runs = 2
start_seed = 0
count = 0
loss_type = 'l2' #['kl', 'l2']
dir_aux = True
ema_alpha = 0.999

lst_datasets = [
    #('seq-cifar10', best_params_seqcif10),
    ('seq-cifar100-aug', best_params_seqcif100),
    #('seq-tinyimg', best_params_seqtiny),
]

p_lst = [0.1, 0.3, 0.5, 0.7]

for seed in range(start_seed, start_seed + num_runs):
    for dataset, params in lst_datasets:
        for buffer_size in lst_buffer_size:
            train_params = params[buffer_size]
            for ema_update_freq in train_params['ema_update_freq_lst']:
                for loss_wt1 in train_params['loss_wt_lst1']:
                    for loss_wt2 in train_params['loss_wt_lst2']:
                        for loss_wt3 in train_params['loss_wt_lst3']:
                            for loss_wt4 in train_params['loss_wt_lst4']:
                                for p in p_lst:
                                    if loss_wt1 == loss_wt3 and loss_wt2 == loss_wt4:
                                        exp_id = f"ccl-{buffer_size}-{dataset}-l{loss_wt1}{loss_wt2}-up{ema_update_freq}-p{p}-s{seed}"
                                    else:
                                        exp_id = f"ccl-{buffer_size}-{dataset}-l{loss_wt1}{loss_wt2}{loss_wt3}{loss_wt4}-up{ema_update_freq}--p{p}-s{seed}"
                                    job_args = f"python /volumes2/continual_learning/mammoth/mammothssl/main.py  \
                                        --experiment_id {exp_id} \
                                        --seed {seed} \
                                        --model ccl \
                                        --abl_mode nolog \
                                        --dataset {dataset} \
                                        --buffer_size {buffer_size} \
                                        --aux {train_params['aux']} \
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
                                        --output_dir /data/output-ai/shruthi.gowda/continual/ccl_ema_aug \
                                        --loss_type {loss_type} \
                                        --loss_wt {loss_wt1} {loss_wt2} {loss_wt3} {loss_wt4} \
                                        --ema_alpha {ema_alpha} \
                                        --dir_aux \
                                        --buf_aux \
                                        --ema_update_freq {ema_update_freq} \
                                        --tensorboard \
                                        --csv_log \
                                        --aug_prob {p} \
                                        "


                                    count += 1
                                    os.system(job_args)

print('%s jobs counted' % count)

#                            --save_model \
