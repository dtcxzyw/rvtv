import os
import subprocess
import sys
import tqdm
import shutil

rvtv_exec = sys.argv[1]
dataset = sys.argv[2]
output = "output"

if os.path.exists(output):
    shutil.rmtree(output)
os.makedirs(output)

def test(file):
    try:
        name = os.path.basename(file)
        subprocess.check_call([rvtv_exec, file, '--mattr=+m,+a,+f,+d,+c', '-o', os.path.join(output, name)], stderr=subprocess.DEVNULL,stdout=subprocess.DEVNULL)
        return True
    except:
        return False
    
work_list = []
for file in os.listdir(dataset):
    if file.endswith('.ll'):
        path = os.path.join(dataset, file)
        work_list.append(path)

count = 0
for file in tqdm.tqdm(work_list,maxinterval=1,mininterval=0.5,smoothing=0.99):
    if test(file):
        count += 1
    else:
        # pass
        print('Failed', file)
        sys.exit(1)