import os
import subprocess
import sys
import tqdm
import shutil
from multiprocessing import Pool

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
        return (file, True)
    except:
        return (file, False)
    
work_list = []
for file in os.listdir(dataset):
    if file.endswith('.ll'):
        path = os.path.join(dataset, file)
        work_list.append(path)

progress = tqdm.tqdm(work_list,maxinterval=1,mininterval=0.5,smoothing=0.99)
pool = Pool(processes=16)

for file, res in pool.imap_unordered(test, work_list):
    progress.update()
    if res:
        pass
    else:
        # pass
        print('Failed', file)
        sys.exit(1)

progress.close()
