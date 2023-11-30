import os

os.system('kubectl -n cyber-security-gpu get pod | grep shru | grep Error  > jobs.txt')

lst_jobs = []

count = 0

with open('jobs.txt', 'r') as f:

    for line in f:

        #lst_jobs.append((line.split()[0]).split('s-0')[0] + 's-0')
        lst_jobs.append((line.split()[0]).split('sh')[0] + 'shru')

        #print((line.split()[0]).split('shruthi')[0] + 'shruthi')

for job in lst_jobs:

    count += 1

    os.system('kubectl -n cyber-security-gpu delete job %s' % job)

print(f'{count} jobs deleted')

