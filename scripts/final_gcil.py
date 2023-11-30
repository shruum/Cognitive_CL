import yaml
import os

# job = yaml.safe_load(open(r'template.yaml'))
job = yaml.load(open(r'template_gpu4.yaml'))


best_params_gcilcif100 = {
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
        'sobel_upsample': 'True',
        'loss_types': ['l2'],
        'loss_wt_lst1': [0.01, 0.1],
        'loss_wt_lst2': [0.01],
        'ema_update_freq_lst': [0.05, 0.07]
    },
    500: {
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
        'loss_wt_lst1': [0.1],
        'loss_wt_lst2': [0.01],
        'ema_update_freq_lst': [0.05, 0.07]
    },
}

lst_buffer_size = [200, 500]
num_runs = 1
start_seed = 0
count = 0

lst_datasets = [
    ('gcil-cifar100', best_params_gcilcif100),
    ]

gcil_modes = ['unif', 'longtail']
loss_types = ['l2'] #['kl', 'l2']

for seed in range(start_seed, start_seed + num_runs):
    for dataset, params in lst_datasets:
        for buffer_size in lst_buffer_size:
            train_params = params[buffer_size]
            for gcil_mode in gcil_modes:
                for lr in train_params['lr']:
                    for alpha in train_params['alpha']:
                        for beta in train_params['beta']:
                            for ema_update_freq in train_params['ema_update_freq_lst']:
                                for loss_type in train_params['loss_types']:
                                    for loss_wt1, loss_wt2 in zip(train_params['loss_wt_lst1'],train_params['loss_wt_lst2']):
                                        # for loss_wt2 in train_params['loss_wt_lst2']:
                                        exp_id = f"ccl-{buffer_size}-gcil-{gcil_mode}-lr{lr}-a{alpha}-{loss_type}{loss_wt1}{loss_wt2}-up{ema_update_freq}-s{seed}"
                                        job_args = ["-c", f"python /git/main.py  \
                                            --experiment_id {exp_id} \
                                            --seed {seed} \
                                            --model ccl \
                                            --abl_mode nolog \
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
                                            --sobel_upsample {train_params['sobel_upsample']} \
                                            --batch_size {train_params['batch_size']} \
                                            --output_dir /output/ccl_new \
                                            --loss_type {loss_type} \
                                            --loss_wt {loss_wt1} {loss_wt2} {loss_wt1} {loss_wt2}\
                                            --ema_alpha {0.999} \
                                            --ema_update_freq {ema_update_freq} \
                                            --weight_dist {gcil_mode} \
                                            --dir_aux \
                                            --buf_aux \
                                            --tensorboard \
                                            --csv_log \
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
