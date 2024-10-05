import os
import subprocess
import sys
import tqdm
from multiprocessing import Pool

alive_tv = sys.argv[1]
dataset = sys.argv[2]
output = "output"

if not os.path.exists(output):
    exit(1)

skip_list = [
    "52392226991139f2.ll", # inttoptr
    "73ad6f7122c6dc64.ll",
    "65b29ef74a83b809.ll",
    "214b13e71a891c0e.ll",
    "35d50a48463eefb8.ll",
    "28f959323d9104cd.ll",
    "0a7dbeba135d6bc0.ll",
    "ef8634a74e125c37.ll", # gep null
    "b82649d5cbf4ffb0.ll",
    "4c6ffd28bf867774.ll",
    "2e2b686d2720345c.ll",
    "b6129564c2f6209b.ll",
    "c45fec086c501f40.ll",
    "b04f469b12072f44.ll",
    "a9bc88c75f52bb5d.ll",
    "22af518c2b6cb393.ll",
    "9efc627554392b5c.ll",
    "71bede1952caa692.ll",
    "058d472cadba2ed4.ll",
    "e221858f6168f6bb.ll",
    "dbef4917fe3e1c5f.ll",
    "761cb698099d954f.ll",
    "dfd7703ed4d547e0.ll",
    "1e48e11603633f9b.ll",
    "02c84802fc0801eb.ll",
    "811181fd70820aa3.ll",
    "d0adc9f443688c98.ll",
    "0498655192e99e71.ll",
    "cb03909301273330.ll",
    "0fd0c852faafe107.ll", # ptr cmp
    "75a2ca2dbd7ae03a.ll", # 93414
    "a1259b1d1fd47001.ll",
    "c69e9d55100b4c2f.ll",
    "9dcd6ca3467d4ec3.ll",
    "d2186a01f060e6cc.ll",
    "2beeb3f276cd973c.ll",
    "2048af87c3b2aa84.ll"
]

def test(file):
    try:
        name = os.path.basename(file)
        tgt = os.path.join(output, name)
        if not os.path.exists(tgt):
            return ("", True)
        cmd = [alive_tv, '--smt-to=1000', '--tgt-is-asm', '--disable-undef-input', '--disable-poison-input', tgt + ".src", tgt]
        out = subprocess.check_output(cmd).decode('utf-8')
        if "Transformation doesn't verify!" in out:
            return (' '.join(cmd), False)
        if out.count("ERROR: Timeout") != out.count("ERROR:"):
            return (' '.join(cmd), False)
        return ("", True)
    except Exception as e:
        return (' '.join(cmd), False)
    
work_list = []
for file in os.listdir(dataset):
    if file.endswith('.ll'):
        is_skipped = False
        for skip in skip_list:
            if skip in file:
                is_skipped = True
                break
        if is_skipped:
            continue
        path = os.path.join(dataset, file)
        work_list.append(path)

progress = tqdm.tqdm(work_list,maxinterval=1,mininterval=0.5,smoothing=0.99)
pool = Pool(processes=16)

with open("alive2.log", 'w') as log:
    for cmd, res in pool.imap_unordered(test, work_list):
        progress.update()
        if res:
            pass
        else:
            log.write(cmd + '\n')
            log.flush()

progress.close()
