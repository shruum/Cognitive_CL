import yaml
import os

# job = yaml.load(open(r'template.yaml'))
# job = yaml.load(open(r'scripts/template_gpu4.yaml'))

best_params_seqcif10 = {
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

lst_buffer_size = [200, 500] #[200, 500, 5120]
num_runs = 1
start_seed = 0
count = 0
loss_types = ['l2'] #['kl', 'l2']
dir_aux_lst = {True, False}
dir_aux = True

lst_datasets = [
    ('seq-cifar10', best_params_seqcif10),
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
                                exp_id = f"cll-improv-{buffer_size}-{dataset}-l{loss_wt1}-{loss_wt2}-s{seed}"
                                job_args = f"python /volumes2/continual_learning/mammoth/mammothssl/main.py  \
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
                                    --output_dir /data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1_l2 \
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

