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

jobnames = sys.argv[1:]
print(jobnames)
jobname = jobnames[0]

path = '/home/kolb/Divest_Experiments/cluster_tools/'
cmd2 = 'squeue -u kolb'

# check, if we got there
print('starting...')
stdin, stdout, stderr = client.exec_command('ls '+path)
time.sleep(.5)
print(stdout.read().decode('ascii'))
t0 = time.clock()

while True:
    for jobname in jobnames:
        cmd3 = f'sbatch {path}{jobname}.job'
        try:
            stdin, stdout, stderr = client.exec_command(cmd2)
            time.sleep(2)
            rtn = stdout.read().decode('ascii')
            if jobname in rtn:
                print(f'running... {time.clock()-t0}')
            else:
                print(f'{jobname} died... trying to restart')
                print(cmd3)
                print(rtn)
                t0 = time.clock()
                stdin, stdout, stderr = client.exec_command(cmd3)
                time.sleep(10)
                print(stdout.read().decode('ascii'))
        except KeyboardInterrupt:
            print('closing')
            client.close()
            agent.close()
        except TimeoutError:
            print('connection timed out. retry.')


