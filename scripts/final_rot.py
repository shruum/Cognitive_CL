import yaml
import os

job = yaml.safe_load(open(r'template.yaml'))

best_params_rotmnist = {
    200: {
        'idt': 'v1',
        'lr': [0.3],
        'minibatch_size': 128,
        'alpha': 0.3,
        'beta': 0.5,
        'batch_size': 128,
        'n_epochs': 1,
        'aux': 'shape',
        'img_size': 28,
        'shape_filter': 'sobel',
        'shape_upsample_size': [56, 84],
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
        'loss_type': ['kl'],
        'loss_wt_lst1': [0.01],
        'loss_wt_lst2': [0.01],
        'loss_wt_lst3': [0.1],
        'loss_wt_lst4': [0.01],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [0.5, 0.8, 1.0]
    },
    500: {
        'idt': 'v1',
        'lr': [0.3],
        'minibatch_size': 128,
        'alpha': 0.2,
        'beta': 0.5,
        'batch_size': 128,
        'n_epochs': 1,
        'aux': 'shape',
        'img_size': 28,
        'shape_filter': 'sobel',
        'shape_upsample_size': [56, 84],
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
        'loss_type': ['kl'],
        'loss_wt_lst1': [0.01],
        'loss_wt_lst2': [0.1, 0.01],
        'loss_wt_lst3': [0.1],
        'loss_wt_lst4': [0.01],
        'ema_alpha_lst': [0.999],
        'ema_update_freq_lst': [0.5, 0.8, 1.0]
    },
}


lst_buffer_size = [200, 500]
num_runs = 20
start_seed = 0
count = 0
loss_type = 'l2' #['kl', 'l2']
dir_aux = True
ema_alpha = 0.999

lst_datasets = [
    #('seq-cifar10', best_params_seqcif10),
    ('rot-mnist', best_params_rotmnist),
    #('seq-tinyimg', best_params_seqtiny),
]
dname = 'rotm'
for seed in range(start_seed, start_seed + num_runs):
    for dataset, params in lst_datasets:
        for buffer_size in lst_buffer_size:
            train_params = params[buffer_size]
            for lr in train_params['lr']:
                for ema_update_freq in train_params['ema_update_freq_lst']:
                    for shape_upsample_size in train_params['shape_upsample_size']:
                        for loss_wt1 in train_params['loss_wt_lst1']:
                            for loss_wt2 in train_params['loss_wt_lst2']:
                                for loss_wt3 in train_params['loss_wt_lst3']:
                                    for loss_wt4 in train_params['loss_wt_lst4']:
                                        if loss_wt1 == loss_wt3 and loss_wt2 == loss_wt4:
                                            exp_id = f"ccl-{buffer_size}-{dname}-lr-{lr}-l{loss_wt1}{loss_wt2}-up{ema_update_freq}-sob{shape_upsample_size}-s{seed}"
                                        else:
                                            exp_id = f"ccl-{buffer_size}-{dname}-lr-{lr}-l{loss_wt1}{loss_wt2}{loss_wt3}{loss_wt4}-up{ema_update_freq}-sob{shape_upsample_size}-s{seed}"
                                        job_args = ["-c", f"python /git/main.py  \
                                            --experiment_id {exp_id} \
                                            --seed {seed} \
                                            --model ccl \
                                            --abl_mode nolog \
                                            --dataset {dataset} \
                                            --buffer_size {buffer_size} \
                                            --aux {train_params['aux']} \
                                            --lr {lr} \
                                            --minibatch_size {train_params['minibatch_size']} \
                                            --n_epochs {train_params['n_epochs']} \
                                            --img_size {train_params['img_size']} \
                                            --shape_filter {train_params['shape_filter']} \
                                            --shape_upsample_size {shape_upsample_size} \
                                            --sobel_gauss_ksize {train_params['sobel_gauss_ksize']} \
                                            --sobel_ksize {train_params['sobel_ksize']} \
                                            --sobel_upsample {train_params['sobel_upsample']} \
                                            --batch_size {train_params['batch_size']} \
                                            --output_dir /output/ccl_mnist \
                                            --loss_type {loss_type} \
                                            --loss_wt {loss_wt1} {loss_wt2} {loss_wt3} {loss_wt4} \
                                            --ema_alpha {ema_alpha} \
                                            --dir_aux \
                                            --buf_aux \
                                            --ema_update_freq {ema_update_freq} \
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

    #                            --save_model \
