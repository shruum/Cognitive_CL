import yaml
import os

# job = yaml.load(open(r'template.yaml'))
job = yaml.load(open(r'template_gpu4.yaml'))

best_params_seqcif10 = {
    200: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha': 0.1,
        'beta': 0.5,
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
    },
    500: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha': 0.2,
        'beta': 0.5,
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
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
        'img_size': 32,
    },
    500: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha': 0.2,
        'beta': 0.5,
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
    },
}

best_params_seqtiny = {
    200: {'lr': 0.03,
          'minibatch_size': 32,
          'softmax_temp': 2.0,
          'alpha': 0.1,
          'beta': 1.0,
          'batch_size': 32,
          'img_size': 64,
          'n_epochs': 100},
    500: {'lr': 0.03,
          'minibatch_size': 32,
          'alpha': 0.2,
          'beta': 0.5,
          'batch_size': 32,
          'img_size': 64,
          'n_epochs': 100},
}

best_params_domain = {
    200: {'lr': 0.05,
          'minibatch_size': 16,
          'alpha': 0.1,
          'beta': 1.0,
          'batch_size': 16,
          'img_size': 112,
          'n_epochs': 50},
    500: {'lr': 0.05,
          'minibatch_size': 16,
          'alpha': 0.2,
          'beta': 0.5,
          'batch_size': 16,
          'img_size': 112,
          'n_epochs': 50},
}

lst_buffer_size = [200, 500]
num_runs = 3
start_seed = 0
count = 0
p = [0.1, 0.2, 0.3, 0.4, 0.5. 0.6, 0.7, 0.]

lst_datasets = [
     # ('seq-cifar10', best_params_seqcif10),
     # ('seq-cifar100', best_params_seqcif100),
     #('seq-tinyimg', best_params_seqtiny),
     # ('domain-net', best_params_domain),
    ('seq-cifar100', best_params_seqcif100),
]
for seed in range(start_seed, start_seed + num_runs):
    for dataset, params in lst_datasets:
        for buffer_size in lst_buffer_size:
            train_params = params[buffer_size]
            exp_id = f"derpp-base-aug-model-{buffer_size}-{dataset}-s{seed}"
            job_args = ["-c", f"python /git/main.py  \
                --experiment_id {exp_id} \
                --seed {seed} \
                --model derpp \
                --dataset {dataset} \
                --buffer_size {buffer_size} \
                --alpha {train_params['alpha']} \
                --beta {train_params['beta']} \
                --lr {train_params['lr']} \
                --minibatch_size {train_params['minibatch_size']} \
                --n_epochs {train_params['n_epochs']} \
                --batch_size {train_params['batch_size']} \
                --output_dir /output/base_aug \
                --tensorboard \
                --csv_log \
                --aug_norm \
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

