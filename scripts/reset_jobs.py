import os
os.system('kubectl -n cyber-security-gpu get pod | grep shru | grep Error > pending_jobs.txt')
lst_jobs = []
count = 0
with open('pending_jobs.txt', 'r') as f:
    for line in f:
        lst_jobs.append(line.split()[0])
for job in lst_jobs:
    count += 1
    os.system('kubectl -n cyber-security-gpu delete pod %s' % job)
print(f'{count} jobs deleted')