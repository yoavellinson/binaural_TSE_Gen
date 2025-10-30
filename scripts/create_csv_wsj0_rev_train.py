from pathlib import Path
import random
from glob import glob
from tqdm import tqdm
import random
from random import shuffle
import pandas as pd
import numpy as np
import sys
import pickle
from reverbration_simulator.create_setup import run_single_simulation
import torch
def load_db(path):
    data = torch.load(path)
    return {'patches': data['patches'], 'pos': data['pos']}

root_path = Path('/dsi/gannot-lab/gannot-lab1/datasets/sharon_db/wsj0/Train/') #lrs2
root_sofas_path = Path('/home/workspace/yoavellinson/binaural_TSE_Gen/sofas')
root_pts_path = Path('/home/workspace/yoavellinson/binaural_TSE_Gen/pts')

lrs_ids_paths = glob(str(root_path/'**/*.wav'))
output_csv = Path("/home/workspace/yoavellinson/binaural_TSE_Gen/csvs/HRTF_train_VAE_wsj0_10k.csv")

df = pd.DataFrame(columns=["sofa_path","pt_path","speaker_1","az_1","elev_1","hrir_rev_1_path","hrir_zero_1_path","speaker_2","az_2","elev_2","hrir_rev_2_path","hrir_zero_2_path","rt_60","sir"])

with open("/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/sofa_az_elev_lookup.pkl", "rb") as f:
    az_elev_lookup = pickle.load(f)

speakers ={}
sofas = {}
speakers_list = glob(str(root_path/'**'))
sofa_db_list = glob(str(root_sofas_path/'**'))
j=0
max_samples = 10000

for i in tqdm(range(10)):
    for speaker in speakers_list:
        name = speaker.split('/')[-1]
        l = glob(speaker+'/*.wav')
        if len(l)>1:
            shuffle(l)
            speakers[name] = l

    for db in sofa_db_list:
        name = db.split('/')[-1]
        l = glob(db+'/*.sofa')
        if len(l)>1:
            shuffle(l)
            sofas[name] = l
    
    while len(speakers.keys()) >=2 and j<max_samples:
        # try:
        speaker = random.sample(list(speakers.keys()),1)[0]
        sofa_db = random.sample(list(sofas.keys()),1)[0]
        sofa_path = sofas[sofa_db].pop()
        s1 = speakers[speaker].pop()
        pt_path = sofa_path.replace('sofas','pts').replace('.sofa','.pt')
        if len(speakers[speaker])<1:
            speakers.pop(speaker)
        rnd_speaker = random.sample(list(speakers.keys()),1)[0]
        while rnd_speaker == speaker:
            rnd_speaker = random.sample(list(speakers.keys()),1)[0]
        s2 = speakers[rnd_speaker].pop()
        if len(speakers[rnd_speaker])<1:
            speakers.pop(rnd_speaker)

        d = load_db(pt_path)
        valid_pos = d['pos']

        az1 = float(random.choice(valid_pos[:,0]))
        valid_az = [az for az in valid_pos[:,0] if abs(float(az)-float(az1)) >= 30]
        az2 = float(random.choice(valid_az))#random.choice(list(range(0, 91)) + list(range(270, 361)))
        valid_elev_1 = [elev for elev in valid_pos[valid_pos[:,0]==az1][:,1] if abs(elev)<30]

        while valid_elev_1==[]: # if elev is not avilable in pos
            az1 = float(random.choice(valid_pos[:,0]))
            valid_az = [az for az in valid_pos[:,0] if abs(float(az)-float(az1)) >= 30]
            az2 = float(random.choice(valid_az))#random.choice(list(range(0, 91)) + list(range(270, 361)))
            valid_elev_1 = [elev for elev in valid_pos[valid_pos[:,0]==az1][:,1] if abs(elev)<20]
        elev1 = float(random.choice(valid_elev_1))
        
        valid_elev_2 = [elev for elev in valid_pos[valid_pos[:,0]==az2][:,1] if abs(elev)<20]
        while valid_elev_2 ==[]: # if elev is not avilable in pos
            az2 = float(random.choice(valid_az))#random.choice(list(range(0, 91)) + list(range(270, 361)))
            valid_elev_1 = [elev for elev in valid_pos[valid_pos[:,0]==az1][:,1] if abs(elev)<20]
        elev2 = float(random.choice(valid_elev_2))

        az1= float(f'{float(az1):.3f}')
        az2= float(f'{float(az2):.3f}')
        elev1 = float(f'{float(elev1):.3f}')
        elev2 = float(f'{float(elev2):.3f}')

        sir = np.random.uniform(-5,5, 1)[0]

        h1_rev,h1_zero,h2_rev,h2_zero,rt_60 =run_single_simulation(j,sofa_path,az1,elev1,az2,elev2,dir_add_str='_train_10k')
        df.loc[len(df)] = [sofa_path,pt_path,s1,az1,elev1,h1_rev,h1_zero,s2,az2,elev2,h2_rev,h2_zero,rt_60,sir]
        j+=1
        # except:
        #     continue
        df.to_csv(output_csv)

df.to_csv(output_csv)
