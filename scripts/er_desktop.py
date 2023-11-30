import yaml
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# job = yaml.load(open(r'template.yaml'))
# job = yaml.load(open(r'scripts/template_gpu4.yaml'))

best_params_seqcif10 = {
    200: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha_mm': [0.3],
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
        'aux': 'shape',
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
    },
    500: {
        'idt': 'v1',
        'lr': 0.1,
        'minibatch_size': 32,
        'alpha_mm': [0.3],
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
        'aux': 'shape',
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
      },
    5120: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha_mm': [0.3],
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
        'aux': 'shape',
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
      }
}

best_params_seqcif100 = {
    200:{'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha_mm': [0.3],
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
        'aux': 'shape',
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
         },
   500: {'idt': 'v1',
        'lr': 0.1,
        'minibatch_size': 32,
        'alpha_mm': [0.3],
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
        'aux': 'shape',
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
         },
   5120: {'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha_mm': [0.3],
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
        'aux': 'shape',
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
        }
}

best_params_seqtiny = {
    200: {'lr': 0.03,
          'minibatch_size': 32,
          'softmax_temp': 2.0,
          'alpha': 0.1,
          'batch_size': 32,
          'n_epochs': 100,
          'aux': 'shape',
          'shape_filter': 'sobel',
          'shape_upsample_size': 128,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': 'True',
          'loss_type': ['kl'],
          },
    500: {'lr': 0.03,
          'minibatch_size': 32,
          'alpha': 0.1,
          'batch_size': 32,
          'n_epochs': 100,
          'aux': 'shape',
          'shape_filter': 'sobel',
          'shape_upsample_size': 128,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': 'True',
          'loss_type': ['kl'],
          },
    5120: {'lr': 0.03,
           'minibatch_size': 32,
           'alpha': 0.1,
           'batch_size': 32,
           'n_epochs': 100,
           'aux': 'shape',
           'shape_filter': 'sobel',
           'shape_upsample_size': 128,
           'sobel_gauss_ksize': 3,
           'sobel_ksize': 3,
           'sobel_upsample': 'True',
           'loss_type': ['kl'],
           }
}

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


loss_wt_lst1 =  [1.0] #[0.001, 0.01, 0.1] #
loss_wt_lst2 = [1.0] #[0.001, 0.01, 0.1] #
lst_buffer_size = [200, 500] #, 5120]
num_runs = 2
start_seed = 0
count = 0

datasets = ['seq-cifar10', 'seq-cifar100']
loss_types = ['l2']# ['kl'] #, 'l2']

lst_datasets = [
    # ('seq-cifar10', best_params_seqcif10),
    # ('seq-cifar100', best_params_seqcif100),
    ('domain-net', best_params_domain),
    # ('seq-tinyimg', best_params_seqtiny),
    ]
alpha_derp = 0.1
beta_derp = 0.5

for seed in range(start_seed, start_seed + num_runs):
    for dataset, params in lst_datasets:
        for buffer_size in lst_buffer_size:
            train_params = params[buffer_size]
            # for alpha1, alpha2 in zip(lst_alpha_der1, lst_alpha_der2):
            # for loss_type in loss_types:
            #     for loss_wt1 in loss_wt_lst1:
            #         for loss_wt2 in loss_wt_lst2:
            exp_id = f"er-dual-{buffer_size}-{dataset}-s{seed}"
            job_args = f"python /volumes2/continual_learning/mammoth/mammothssl/main.py  \
                --experiment_id {exp_id} \
                --seed {seed} \
                --model er \
                --dataset {dataset} \
                --buffer_size {buffer_size} \
                --aux {train_params['aux']} \
                --lr {train_params['lr']} \
                --minibatch_size {train_params['minibatch_size']} \
                --n_epochs {train_params['n_epochs']} \
                --shape_filter {train_params['shape_filter']} \
                --shape_upsample_size {train_params['shape_upsample_size']} \
                --sobel_gauss_ksize {train_params['sobel_gauss_ksize']} \
                --sobel_ksize {train_params['sobel_ksize']} \
                --sobel_upsample {train_params['sobel_upsample']} \
                --batch_size {train_params['batch_size']} \
                --output_dir /data/output-ai/shruthi.gowda/continual/er_dual \
                --tensorboard \
                --csv_log \
                --data_combine \
                --img_size {train_params['img_size']} \
                "
            # # set job params
            # job['metadata']['name'] = exp_id + '-shru'
            # job['spec']['template']['spec']['containers'][0]['args'] = job_args
            #
            # yaml_out = 'temp/%s.yaml' % exp_id
            #
            # with open(yaml_out, 'w') as outfile:
            #     yaml.dump(job, outfile, default_flow_style=False)

            count += 1
            os.system(job_args)

print('%s jobs counted' % count)

