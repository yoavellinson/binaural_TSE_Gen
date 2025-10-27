import numpy as np
from abs_coeff import absorption_data
import random
from pathlib import Path
import os
import subprocess

sind = lambda degrees: np.sin(np.deg2rad(degrees))
cosd = lambda degrees: np.cos(np.deg2rad(degrees))
tand = lambda degrees: np.tan(np.deg2rad(degrees))


def generate_A(absorption_data):
    walls = list(absorption_data['walls'].keys())
    ceil =  list(absorption_data['ceilings'].keys())
    floors =  list(absorption_data['floors'].keys())

    A = np.zeros((6,6))
    for i in range(4):
        A[i,:] = absorption_data['walls'][random.choice(walls)]
    A[4,:] = absorption_data['ceilings'][random.choice(ceil)]
    A[5,:] = absorption_data['floors'][random.choice(floors)]
    return A

def compute_surface_area(Lx,Ly,Lz,A):
    A = A.T
    V = Lx * Ly * Lz
    wall_xz = Lx * Lz
    wall_yz = Ly * Lz
    wall_xy = Lx * Ly
    S = (wall_yz * (A[:, 0] + A[:, 1]) +
        wall_xz * (A[:, 2] + A[:, 3]) +
        wall_xy * (A[:, 4] + A[:, 5]))
    return S,V

def gen_rt60(x,y,z):
    A = generate_A(absorption_data)
    S,V = compute_surface_area(x,y,z,A)
    RT60 = (55.25 / 343) * V / S
    fb_rt60 = np.mean(RT60[2:])
    return A,fb_rt60,RT60

def az_elev_to_xyz_yaw_pitch(az,elev,theta,x,y,z):
    r = 1.5 + np.random.randn()*0.2
    R = np.array([
    [cosd(theta), sind(theta)],
    [-sind(theta),  cosd(theta)]
    ])
    if (270<= az<=360):
        az -= 360
    if az>=0:
        yaw = -90 - abs(abs(-90) -abs(az)) + theta
    else:
        yaw = 180 +az + theta
    xy = np.array([[r*cosd(az),r*sind(az)]])
    xy = (xy@R)[0]
    delta_z = r*tand(elev)
    pitch = -elev
    return x+xy[0],y+xy[1],z+delta_z,yaw,pitch

def run_single_simulation(j,sofa_path,az1,elev1,az2,elev2):
    room_x = 10
    room_y = 10
    room_z = 2.8
    z_head = np.random.normal(1.685,0.083)

    z_speaker = 1.6
    r = 3
    print(f"[{j}] Starting simulation", flush=True)
    min_rt60 = 0.1
    max_rt60 = np.random.uniform(min_rt60,0.8)
    
    x,y = np.random.uniform(r, room_x - r),np.random.uniform(r, room_y - r)
    out_file_name = f'h_rt60'
    out_file_name_order_0 = f'h_first'
    sofa_path_obj = Path(sofa_path)
    name = f'{sofa_path_obj.parent.stem}_{sofa_path_obj.stem}_az1{az1}_elev1{elev1}_az2{az2}_elev2{elev2}'
    out_dir =Path(f'/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf/{name}')

    out_dir.mkdir(exist_ok=True)

    theta = np.random.uniform(0,90)
    R = np.array([
        [cosd(theta), sind(theta)],
        [-sind(theta),  cosd(theta)]
    ])


    A,fb_rt60,RT60 = gen_rt60(room_x,room_y,room_z)
    while fb_rt60 > max_rt60 and fb_rt60<min_rt60:
        A,fb_rt60,RT60 = gen_rt60(room_x,room_y,room_z)
 
    with open(f'/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/clean_template.m','r') as f:
        template = f.read()

    receiver = f"receiver(1).description      = 'SOFA {sofa_path} interp=1 norm=1 resampling=1';\n"
    template +=receiver

    new_dir = out_dir/f'rt_60_{fb_rt60}'
    new_dir.mkdir(exist_ok=True)
    room_surface_absorption = f'''room.surface.absorption  = [{A[0,0]} {A[0,1]} {A[0,2]} {A[0,3]} {A[0,4]} {A[0,5]};
                                    {A[1,0]} {A[1,1]} {A[1,2]} {A[1,3]} {A[1,4]} {A[1,5]};
                                    {A[2,0]} {A[2,1]} {A[2,2]} {A[2,3]} {A[2,4]} {A[2,5]};
                                    {A[3,0]} {A[3,1]} {A[3,2]} {A[3,3]} {A[3,4]} {A[3,5]};
                                    {A[4,0]} {A[4,1]} {A[4,2]} {A[4,3]} {A[4,4]} {A[4,5]};
                                    {A[5,0]} {A[5,1]} {A[5,2]} {A[5,3]} {A[5,4]} {A[5,5]}];\n'''
    
    template+=room_surface_absorption
    receiver_orientation = f"receiver(1).orientation      = [ {theta} 0 0 ];\n"
    room_dimensions = f'room.dimension              = [ {room_x} {room_y} {room_z} ];\n'
    receiver_location = f"receiver(1).location         = [ {x} {y} {z_head} ]; \n"
    template+=receiver_orientation+receiver_location+room_dimensions+'\n'

    # for i,d in azs.items():
    #     i = int(i)
    x1,y1,z1,yaw1,pitch1 = az_elev_to_xyz_yaw_pitch(az1,elev1,theta,x,y,z_head)

    sl = f"source({1}).location           = [ {x1} {y1} {z1} ];\n"
    so =f"source({1}).orientation        = [ {yaw1} {pitch1} 0 ];\n"     
    sd =f"source({1}).description        = 'subcardioid';\n"
    template+=sl+so+sd

    x2,y2,z2,yaw2,pitch2 = az_elev_to_xyz_yaw_pitch(az2,elev2,theta,x,y,z_head)

    sl = f"source({2}).location           = [ {x2} {y2} {z2} ];\n"
    so =f"source({2}).orientation        = [ {yaw2} {pitch2} 0 ];\n"     
    sd =f"source({2}).description        = 'subcardioid';\n"
    template+=sl+so+sd

    template_order_0 = template
    order = 10
    order_line = f'options.reflectionorder     = [ {order} {order} {order} ];\n'
    options_outputname = f"options.outputname			= '{new_dir/out_file_name}';\n"

    template +=order_line+options_outputname

    order =0
    order_line_0 = f'options.reflectionorder     = [ {order} {order} {order} ];\n'
    options_outputname_order_0 = f"options.outputname			= '{new_dir/out_file_name_order_0}';\n"
    template_order_0+=order_line_0+options_outputname_order_0
    
    filename_order_0 = f'/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/setup_files/clean_random_{j}_new_order_0.m'
    with open(filename_order_0, 'w') as f:
        f.write(template_order_0)

    filename = f'/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/setup_files/clean_random_{j}_new.m'
    with open(filename, 'w') as f:
        f.write(template)

    sim_command = f'/home/workspace/yoavellinson/sofamyroom/build/sofamyroom {filename}'
    result = subprocess.run(sim_command, shell=True)
    if result.returncode != 0:
        print(f"[{j}] Subprocess failed with code {result.returncode}", flush=True)
    sim_command=f'/home/workspace/yoavellinson/sofamyroom/build/sofamyroom {filename_order_0}'
    result = subprocess.run(sim_command, shell=True)
    if result.returncode != 0:
        print(f"[{j}] Subprocess failed with code {result.returncode}", flush=True)
    
    h1_rev = new_dir/f'{out_file_name}_receiver_0.wav'
    h1_zero = new_dir/f'{out_file_name_order_0}_receiver_0.wav'
    h2_rev = new_dir/f'{out_file_name}_receiver_1.wav'
    h2_zero = new_dir/f'{out_file_name_order_0}_receiver_1.wav'
    return h1_rev,h1_zero,h2_rev,h2_zero,fb_rt60

if __name__ == "__main__":
    # from multiprocessing import Pool
    # with Pool(processes=1) as pool:
    #     pool.map(run_single_simulation, range(1))
    run_single_simulation(j=0,sofa_path='/home/workspace/yoavellinson/binaural_TSE_Gen/sofas/viking/subj_A.sofa',az1=90,elev1=10,az2=-90,elev2=-10)
