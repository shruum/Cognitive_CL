import yaml
import os

job = yaml.safe_load(open(r'template.yaml'))
# job = yaml.load(open(r'template_gpu4.yaml'))

best_params_seqcif10 = {
    200: {
        'idt': 'v1',
        'lr': [0.03],
        'minibatch_size': 32,
        'alpha': [0.1],
        'beta': [0.5],
        'batch_size': 32,
        'n_epochs': 50,
        'aux': 'shape',
        'img_size': 32,
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
        'loss_types': ['l2'],
        'loss_wt_lst1': [0.1, 0.01],
        'loss_wt_lst2': [0.1, 0.01],
        'ema_update_freq_lst': [0.1, 0.2]
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
        'loss_types': ['l2'],
        'loss_wt_lst1': [0.1],
        'loss_wt_lst2': [0.1],
        'ema_update_freq_lst': [0.2, 0.1]
    },
}

best_params_seqcif100 = {
    200: {
        'idt': 'v1',
        'lr': [0.1, 0.03],
        'minibatch_size': 32,
        'alpha': [0.1, 0.15],
        'beta': [0.5],
        'batch_size': 32,
        'n_epochs': 50,
        'aux': 'shape',
        'img_size': 32,
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': ['True'],
        'loss_types': ['l2'],
        'loss_wt_lst1': [0.1],
        'loss_wt_lst2': [0.01],
        'ema_update_freq_lst': [0.04, 0.05, 0.06, 0.07, 0.09]
    },
    500: {
        'idt': 'v1',
        'lr': [0.1, 0.03],
        'minibatch_size': 32,
        'alpha': [0.15, 0.2],
        'beta': [0.5, 1.0],
        'batch_size': 32,
        'n_epochs': 50,
        'aux': 'shape',
        'img_size': 32,
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': ['True'],
        'loss_types': ['l2'],
        'loss_wt_lst1': [0.1],
        'loss_wt_lst2': [0.01],
        'ema_update_freq_lst': [0.06, 0.07, 0.08, 0.09, 0.1]
    },
}

best_params_seqcif100_single = {
    200: {
        'idt': 'v1',
        'lr': [0.03],
        'minibatch_size': 32,
        'alpha': [0.15],
        'beta': [0.5],
        'batch_size': 32,
        'n_epochs': 50,
        'aux': 'shape',
        'img_size': 32,
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': ['True'],
        'loss_types': ['l2'],
        'loss_wt_lst1': [0.1],
        'loss_wt_lst2': [0.01],
        'ema_update_freq_lst': [0.06]
    },
    500: {
        'idt': 'v1',
        'lr': [0.03],
        'minibatch_size': 32,
        'alpha': [0.15],
        'beta': [0.5],
        'batch_size': 32,
        'n_epochs': 50,
        'aux': 'shape',
        'img_size': 32,
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': ['True'],
        'loss_types': ['l2'],
        'loss_wt_lst1': [0.1],
        'loss_wt_lst2': [0.01],
        'ema_update_freq_lst': [0.09]
    },
}

best_params_seqtiny = {
    200: {'lr': [0.02, 0.04, 0.06],
          'minibatch_size': 32,
          'alpha': [0.1],
          'beta': [0.5, 1.0],
          'batch_size': 32,
          'n_epochs': 50,
          'aux': 'shape',
          'img_size': 64,
          'shape_filter': 'sobel',
          'shape_upsample_size': 128,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': ['True', 'False'],
          'loss_types': ['l2'],
          'loss_wt_lst1': [0.1],
          'loss_wt_lst2': [0.1],
          'ema_update_freq_lst': [0.05, 0.06, 0.07]
          },
    500: {'lr': [0.02, 0.03],
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
          'loss_wt_lst1': [0.1, 0.2, 0.1],
          'loss_wt_lst2': [0.1, 0.1, 0.2],
          'ema_update_freq_lst': [0.06, 0.07, 0.08]
          },
}

best_params_seqtiny_single = {
    200: {'lr': [0.03],
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
          'loss_wt_lst1': [0.1],
          'loss_wt_lst2': [0.1, 0.01],
          'ema_update_freq_lst': [0.06, 0.08]
          },
    500: {'lr': [0.02, 0.03],
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
          'loss_wt_lst1': [0.1, 0.2, 0.1],
          'loss_wt_lst2': [0.1, 0.1, 0.2],
          'ema_update_freq_lst': [0.06, 0.07, 0.08]
          },
}

best_params_domain = {
    200: {'lr': 0.05,
          'minibatch_size': 16,
          'alpha': 0.1,
          'beta': 1.0,
          'batch_size': 16,
          'n_epochs': 50,
          'aux': 'shape',
          'img_size': 112,
          'shape_filter': 'sobel',
          'shape_upsample_size': 224,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': 'False',
          'loss_types': ['kl', 'l2'],
          'loss_wt_lst1': [0.1, 0.01],
          'loss_wt_lst2': [0.1, 0.01],
          'ema_alpha_lst': [0.999],
          'ema_update_freq_lst': [0.06, 0.08]
          },
    500: {'lr': 0.05,
          'minibatch_size': 16,
          'alpha': 0.2,
          'beta': 0.5,
          'batch_size': 16,
          'n_epochs': 50,
          'aux': 'shape',
          'img_size': 112,
          'shape_filter': 'sobel',
          'shape_upsample_size': 128,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': 'False',
          'loss_types': ['kl', 'l2'],
          'loss_wt_lst1': [0.1, 0.01],
          'loss_wt_lst2': [0.1, 0.01],
          'ema_alpha_lst': [0.999],
          'ema_update_freq_lst':[0.06, 0.08]
          },
}

lst_buffer_size = [200] # 500]
num_runs = 3
start_seed = 0
count = 0

lst_datasets = [
     #('seq-cifar10', best_params_seqcif10),
     # ('seq-cifar100', best_params_seqcif100_single),
    #('domain-net', best_params_domain),
    ('seq-tinyimg', best_params_seqtiny_single),
    ]
loss_types = ['l2'] #['kl', 'l2']

for seed in range(start_seed, start_seed + num_runs):
    for dataset, params in lst_datasets:
        for buffer_size in lst_buffer_size:
            train_params = params[buffer_size]
            for lr in train_params['lr']:
                for alpha in train_params['alpha']:
                    for beta in train_params['beta']:
                        for ema_update_freq in train_params['ema_update_freq_lst']:
                            for loss_type in train_params['loss_types']:
                                for loss_wt1, loss_wt2 in zip(train_params['loss_wt_lst1'],train_params['loss_wt_lst2']):
                                    for upsamp in train_params['sobel_upsample']:
                                        if upsamp == 'True':
                                            up = 't'
                                        else:
                                            up = 'f'
                                        # for loss_wt2 in train_params['loss_wt_lst2']:
                                        exp_id = f"cll-tiny{buffer_size}-a{alpha}b{beta}-lr{lr}-{loss_type}{loss_wt1}{loss_wt2}-up{ema_update_freq}-s{seed}"
                                        job_args = ["-c", f"python /git/main.py  \
                                            --experiment_id {exp_id} \
                                            --seed {seed} \
                                            --model derpp_mm_eman1 \
                                            --dataset {dataset} \
                                            --buffer_size {buffer_size} \
                                            --aux {train_params['aux']} \
                                            --alpha_mm {alpha} {alpha} \
                                            --beta_mm {beta} {beta} \
                                            --lr {lr} \
                                            --minibatch_size {train_params['minibatch_size']} \
                                            --n_epochs {train_params['n_epochs']} \
                                            --img_size {train_params['img_size']} \
                                            --shape_filter {train_params['shape_filter']} \
                                            --shape_upsample_size {train_params['shape_upsample_size']} \
                                            --sobel_gauss_ksize {train_params['sobel_gauss_ksize']} \
                                            --sobel_ksize {train_params['sobel_ksize']} \
                                            --sobel_upsample {upsamp} \
                                            --batch_size {train_params['batch_size']} \
                                            --output_dir /output/method/clmm_derpp_v2n1_tiny \
                                            --loss_type {loss_type} \
                                            --loss_wt {loss_wt1} {loss_wt2} \
                                            --ema_alpha {0.999} \
                                            --ema_update_freq {ema_update_freq} \
                                            --dir_aux \
                                            --tensorboard \
                                            --csv_log \
                                            --save_model \
                                            "]
                                        # set job params
                                        job['metadata']['name'] = exp_id + '-shru'
                                        job['spec']['template']['spec']['containers'][0]['args'] = job_args

                                        yaml_out = 'temp/%s.yaml' % exp_id

                                        with open(yaml_out, 'w') as outfile:
                                            yaml.dump(job, outfile, default_flow_style=False)

                                        count += 1
                                        os.system('kubectl -n arl create -f %s' % yaml_out)

print('%s jobs counted' % count)

# --save_model \
