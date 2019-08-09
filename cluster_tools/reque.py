import sys
import time
import subprocess
import paramiko

hostname = 'cluster.pik-potsdam.de'
port = None
username = 'kolb'
password = 'ja_Ko!Lb'

agent = paramiko.agent.Agent()
keys = agent.get_keys()

client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect(hostname, username='kolb', look_for_keys=False)

jobname = sys.argv[1]

path = '/home/kolb/Divest_Experiments/cluster_tools/'
cmd2 = 'squeue -u kolb'
cmd3 = f'sbatch {path}{jobname}.job'

# check, if we got there
print('starting...')
stdin, stdout, stderr = client.exec_command('ls '+path)
time.sleep(.5)
print(stdout.read().decode('ascii'))

while True:
    try:
        stdin, stdout, stderr = client.exec_command(cmd2)
        time.sleep(2)
        rtn = stdout.read().decode('ascii')
        if jobname in rtn:
            print(f'running... {time.clock()}')
        else:
            print('died... trying to restart')
            print(rtn)
            stdin, stdout, stderr = client.exec_command(cmd3)
            time.sleep(10)
            print(stdout.read().decode('ascii'))
    except KeyboardInterrupt:
        print('closing')
        client.close()
        agent.close()


