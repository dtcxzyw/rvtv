import os
import subprocess
import sys
import tqdm

alive_tv = sys.argv[1]
dataset = sys.argv[2]
output = "output"

if not os.path.exists(output):
    exit(1)

def test(file):
    try:
        name = os.path.basename(file)
        tgt = os.path.join(output, name)
        cmd = [alive_tv, '--smt-to=100', '--disable-undef-input', '--disable-poison-input', tgt + ".src", tgt]
        out = subprocess.check_output(cmd).decode('utf-8')
        if "Transformation doesn't verify!" in out:
            print(' '.join(cmd))
            return False
        if out.count("ERROR: Timeout") != out.count("ERROR:"):
            print(' '.join(cmd))
            return False
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
