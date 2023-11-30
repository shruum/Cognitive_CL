import yaml
import os

job = yaml.safe_load(open(r'template.yaml'))
#job = yaml.load(open(r'template_gpu4.yaml'))

best_params_seqcif10 = {
    200: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha': 0.1,
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
        'loss_wt_kl': [0.1, 0.1],
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
        'loss_wt_kl': [0.1, 0.1],
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
        'n_epochs': 50,
        'aux': 'shape',
        'img_size': 32,
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
        'loss_type': ['kl'],
        'loss_wt_kl': [0.1, 0.1],
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
        'loss_wt_kl': [0.1, 0.1],
      },
}

best_params_seqtiny = {
    200: {'lr': 0.03,
          'minibatch_size': 32,
          'softmax_temp': 2.0,
          'alpha': 0.1,
          'beta': 1.0,
          'batch_size': 32,
          'n_epochs': 100,
          'aux': 'shape',
          'img_size': 64,
          'shape_filter': 'sobel',
          'shape_upsample_size': 128,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': 'True',
          'loss_type': ['kl'],
          },
    500: {'lr': 0.03,
          'minibatch_size': 32,
          'alpha': 0.2,
          'beta': 0.5,
          'batch_size': 32,
          'n_epochs': 100,
          'aux': 'shape',
          'img_size': 64,
          'shape_filter': 'sobel',
          'shape_upsample_size': 128,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': 'True',
          'loss_type': ['kl'],
          },
}


loss_wt_lst1 = [0.01, 0.01, 0.01, 0.1, 1.0, ] #, 1.0, 10.0, 50.0]
loss_wt_lst2 = [1.0,  0.1,  0.01, 0.1, 0.01, ] #, 1.0, 10.0, 50.0]
lst_buffer_size = [200, 500] #, 500] #, 500] #[200, 500] #, 5120]
num_runs = 1
start_seed = 0
count = 0

lst_ema_update_freq = [0.2, 0.4, 0.8] #[0.2, 0.4, 0.6] #, 0.6] #[0.2, 0.4, 0.6, 0.8]
lst_ema_update_freq_tiny = [0.08, 0.2]
ema_alpha_lst = [0.999] #[0.9, 0.999]
lst_datasets = [
     ('seq-cifar10', best_params_seqcif10),
     ('seq-cifar100', best_params_seqcif100),
     #('seq-tinyimg', best_params_seqtiny),
    ]
loss_types = ['kl', 'l2']
dir_aux = True

for seed in range(start_seed, start_seed + num_runs):
    for dataset, params in lst_datasets:
        for buffer_size in lst_buffer_size:
            train_params = params[buffer_size]
            for ema_update_freq in lst_ema_update_freq:
                for ema_alpha in ema_alpha_lst:
                    for loss_type in loss_types:
                        for loss_wt1, loss_wt2 in zip(loss_wt_lst1,loss_wt_lst2):
                            # for loss_wt2 in loss_wt_lst2:
                            exp_id = f"derppema-v31-{buffer_size}-{dataset}-{loss_type}{loss_wt1}-{loss_wt2}-up{ema_update_freq}-s{seed}"
                            job_args = ["-c", f"python /git/main.py  \
                                --experiment_id {exp_id} \
                                --seed {seed} \
                                --model derpp_mm_ema2 \
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
                                --output_dir /output/clmm_derpp_v3_extra \
                                --loss_type {loss_type} \
                                --loss_wt {loss_wt1} {loss_wt2} \
                                --ema_alpha {ema_alpha} \
                                --ema_update_freq {ema_update_freq} \
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

