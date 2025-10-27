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

root_path = Path('/dsi/gannot-lab/gannot-lab1/datasets/sharon_db/wsj0/Train/') #lrs2
root_sofas_path = Path('/home/workspace/yoavellinson/binaural_TSE_Gen/sofas')
root_pts_path = Path('/home/workspace/yoavellinson/binaural_TSE_Gen/pts')

lrs_ids_paths = glob(str(root_path/'**/*.wav'))
output_csv = Path("/home/workspace/yoavellinson/binaural_TSE_Gen/csvs/HRTF_train_VAE_wsj0.csv")

df = pd.DataFrame(columns=["sofa_path","pt_path","speaker_1","az_1","elev_1","hrir_rev_1_path","hrir_zero_1_path","speaker_2","az_2","elev_2","hrir_rev_2_path","hrir_zero_2_path","rt_60","sir"])

with open("/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/sofa_az_elev_lookup.pkl", "rb") as f:
    az_elev_lookup = pickle.load(f)

speakers ={}
sofas = {}
speakers_list = glob(str(root_path/'**'))
sofa_db_list = glob(str(root_sofas_path/'**'))
j=0
max_samples = 5

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

        az1 = random.choice(list(az_elev_lookup[sofa_db].keys()))#random.choice(list(range(0, 91)) + list(range(270, 361)))

        valid_az = [az for az in list(az_elev_lookup[sofa_db].keys()) if abs(float(az)-float(az1)) >= 30]
        az2 = random.choice(valid_az)#random.choice(list(range(0, 91)) + list(range(270, 361)))
        # # if 0 <= az1 <= 90:
        # #     valid_ranges = list(range(0, max(1, az1 - 50))) + list(range(min(90, az1 + 50) + 1, 91))
        # # else:  # 270 <= az1 <= 360
        # #     valid_ranges = list(range(270, max(271, az1 - 50))) + list(range(min(360, az1 + 50) + 1, 361))

        # az2 = random.choice(valid_ranges)
        valid_elev_1 = [elev for elev in list(az_elev_lookup[sofa_db][az1]) if abs(float(elev))<20]
        elev1 = random.choice(valid_elev_1)
        valid_elev_2 = [elev for elev in list(az_elev_lookup[sofa_db][az2]) if abs(float(elev))<20]
        elev2 = random.choice(valid_elev_2)

        az1= float(f'{float(az1):.3f}')
        az2= float(f'{float(az2):.3f}')
        elev1 = float(f'{float(elev1):.3f}')
        elev2 = float(f'{float(elev2):.3f}')

        sir = np.random.uniform(-5,5, 1)[0]

        h1_rev,h1_zero,h2_rev,h2_zero,rt_60 =run_single_simulation(j,sofa_path,az1,elev1,az2,elev2)
        df.loc[len(df)] = [sofa_path,pt_path,s1,az1,elev1,h1_rev,h1_zero,s2,az2,elev2,h2_rev,h2_zero,rt_60,sir]
        j+=1
        # except:
        #     continue

df.to_csv(output_csv)
