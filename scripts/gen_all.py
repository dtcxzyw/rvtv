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

attr = "+m,+a,+f,+d,+c"
attr += ",+zba"
attr += ",+zbb"
attr += ",+zbs"
attr += ",+zbkb"
attr += ",+zicond"
attr += ",+zfa"
attr += ",+zfh"

def test(file):
    name = os.path.basename(file)
    cmd = [rvtv_exec, file, '--mattr=' + attr, '-o', os.path.join(output, name)]
    try:
        subprocess.check_call(cmd, stderr=subprocess.DEVNULL,stdout=subprocess.DEVNULL)
        return ("", True)
    except:
        pass

    return (" ".join(cmd), False)
    
work_list = []
for file in os.listdir(dataset):
    if file.endswith('.ll'):
        path = os.path.join(dataset, file)
        work_list.append(path)

progress = tqdm.tqdm(work_list,maxinterval=1,mininterval=0.5,smoothing=0.99)
pool = Pool(processes=16)

with open("rvtv.log", "w") as log:
    for file, res in pool.imap_unordered(test, work_list):
        progress.update()
        if res:
            pass
        else:
            log.write(file + "\n")
            log.flush()

progress.close()
