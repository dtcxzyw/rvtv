import os
import subprocess
import sys
import tqdm

rvtv_exec = sys.argv[1]
dataset = sys.argv[2]

def test(file):
    try:
        subprocess.check_call([rvtv_exec, file, '--mattr=+m,+a,+f,+d,+c'], stderr=subprocess.DEVNULL,stdout=subprocess.DEVNULL)
        return True
    except:
        return False
    
work_list = []
for file in os.listdir(dataset):
    if file.endswith('.ll'):
        path = os.path.join(dataset, file)
        work_list.append(path)

count = 0
for file in tqdm.tqdm(work_list):
    if test(file):
        count += 1
    else:
        # pass
        print('Failed', file)
        sys.exit(1)

print('Ratio:', count / len(work_list))
