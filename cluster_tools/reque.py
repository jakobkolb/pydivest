import sys
import time
import subprocess

cmd1 = ['squeue', '-u kolb']
cmd2 = ['ls']
cmd3 = ['srun', 'X1a.job']

while True:
    output = subprocess.check_output(cmd1).decode('ascii')
    if 'X1a' not in output:
        print('no')
        rtn = subprocess.check_output(cmd3).decode('ascii')
        print(rtn)
    else:
        print('yes')
    time.sleep(5)

