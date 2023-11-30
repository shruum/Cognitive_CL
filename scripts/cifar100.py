import yaml
import os

job = yaml.safe_load(open(r'template.yaml'))

best_params_seqcif10 = {
    100: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha': 0.3,
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
        'loss_wt_lst1': [0.1],
        'loss_wt_lst2': [0.1],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [0.2]
    },
    150: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha': 0.3,
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
        'loss_wt_lst1': [0.1],
        'loss_wt_lst2': [0.1],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [0.2]
    },
    200: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha': 0.3,
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
        'loss_wt_lst1': [0.1, 0.01, 0.2],
        'loss_wt_lst2': [0.1, 0.01, 0.2],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [0.2]
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
        'loss_wt_lst1': [0.1, 0.01, 0.2],
        'loss_wt_lst2': [0.1, 0.01, 0.2],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [0.2]
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
        'loss_wt_lst1': [0.1],
        'loss_wt_lst2': [0.01],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [0.06, 0.08]
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
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [0.06, 0.08]
    },
}

lst_buffer_size = [100, 150] #[200, 500, 5120]
num_runs = 3
start_seed = 0
count = 0
loss_types = ['l2'] #['kl', 'l2']
dir_aux = True

lst_datasets = [
    ('seq-cifar10', best_params_seqcif10),
    #('seq-cifar100', best_params_seqcif100),

]
n_class = 10
n_task = 10

for seed in range(start_seed, start_seed + num_runs):
    for dataset, params in lst_datasets:
        for buffer_size in lst_buffer_size:
            train_params = params[buffer_size]
            for ema_update_freq in train_params['ema_update_freq_lst']:
                for loss_type in loss_types:
                    for loss_wt1 in train_params['loss_wt_lst1']:
                        for loss_wt2 in train_params['loss_wt_lst2']:
                            exp_id = f"cll-{buffer_size}-{dataset}-l{loss_wt1}-{loss_wt2}-up{ema_update_freq}-t{n_task}s{seed}"
                            job_args = ["-c", f"python /git/main.py  \
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
                                --output_dir /output/method/cifar10_buf \
                                --loss_type {loss_type} \
                                --loss_wt {loss_wt1} {loss_wt2} \
                                --ema_alpha {train_params['ema_alpha_lst'][0]} \
                                --dir_aux \
                                --ema_update_freq {ema_update_freq} \
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

