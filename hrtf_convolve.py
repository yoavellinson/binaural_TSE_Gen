import sys
sys.path.insert(0, '../../src')
import sofa
from scipy import signal
import numpy as np
import soundfile as sf
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.signal import lfilter
import random
# from abs_coeff import absorption_data
from pathlib import Path


sind = lambda degrees: np.sin(np.deg2rad(degrees))
cosd = lambda degrees: np.cos(np.deg2rad(degrees))

class SOFA_HRTF_wrapper:
    def __init__(self,path=''):
        self._load_sofa_path(path)
        self._get_positions()
        
    def _load_sofa_path(self,path):
        #loading the HRTF.sofa file
        try:
            self.hrtf_obj = sofa.Database.open(path)
            self.target_fs = int(self.hrtf_obj.Data.SamplingRate.get_values()[0])
        except:
            if not Path(path).exists():
                raise FileNotFoundError(f"The HRTF.sofa path: '{path}' does not exist.")
    
    def impzest(self,input_signal, output_signal):
        if np.any(np.isnan(input_signal)) or np.any(np.isnan(output_signal)):
            print("Warning: NaN values detected in signals")
        rir_est, _ = signal.deconvolve(output_signal, input_signal)
        return rir_est

    def decimate_rir(self,rir_orig, fs,target_fs=16000):
        rir_orig = np.nan_to_num(rir_orig)
        t = np.linspace(0, 3, int(3 * fs), endpoint=False)
        exc = signal.chirp(t, f0=20, f1=20000, t1=3, method='logarithmic')
        rir_dec = np.zeros_like(rir_orig)
        for i in range(rir_orig.shape[0]):
            exc = np.append(exc, np.zeros(len(rir_orig[i,:]) + 1))

            # Convolve with RIR
            sig = signal.fftconvolve(exc, rir_orig[i,:], mode='full')

            # Decimate the signal
            dec = fs // np.gcd(target_fs, fs)
            exc_dec = signal.decimate(exc, dec)
            exc_dec[np.abs(exc_dec) < 1e-10] = 0

            sig_dec = signal.decimate(sig, dec)
            res = self.impzest(exc_dec, sig_dec)
            rir_dec[i,:len(res)] = res
        return rir_dec
    
    def change_target_fs(self,target_fs):
        if target_fs != self.target_fs:
            self.target_fs = target_fs
        
    def _conv_hrtf_mesurment(self,wav,fs,measurement):
        #convolving a vector 'wav' of time domain audio with an HRTF 
        h = np.zeros([self.hrtf_obj.Dimensions.N,2])
        h[:,0] = self.hrtf_obj.Data.IR.get_values(indices={"M":measurement, "R":0, "E":0})
        h[:,1] = self.hrtf_obj.Data.IR.get_values(indices={"M":measurement, "R":1, "E":0})
        if self.target_fs != fs:
            # h = librosa.core.resample(h.T,orig_sr=self.target_fs,target_sr=fs).T
            h = self.decimate_rir(h.T,self.target_fs,fs).T
        rend_L = signal.fftconvolve(wav.squeeze(),h[:,0])
        rend_R = signal.fftconvolve(wav.squeeze(),h[:,1])
        stereo_audio = np.concatenate((rend_L[:,np.newaxis],rend_R[:,np.newaxis]),axis=1)
        return stereo_audio,h
    
    def _get_positions(self):
        # Summarazing all the available hrtf by azimuth and elevation, assuming R is equal for the whole dataset
        pos = self.hrtf_obj.Source.Position.get_values(system="spherical", angle_unit="degree") #az,elev,r
        self.az_list = np.unique(pos[:,0])
        self.elev_list = np.unique(pos[:,1])
        self.R = np.unique(pos[:,2])[0]
        self.lookup_table = pd.DataFrame(index = self.az_list,columns=self.elev_list )
        for idx,p in enumerate(pos):
            self.lookup_table.at[p[0],p[1]] = idx

    def conv_hrtf_pos(self,wav,fs,az,elev):
        #check if desierd position is available, if not return the closest one
        if az not in self.az_list: 
            az = self.az_list[np.argmin(abs(self.az_list-az))]
        if elev not in self.elev_list:
            elev = self.elev_list[np.argmin(abs(self.elev_list-elev))]
        measurement = self.lookup_table.at[az,elev]
        if np.isnan([measurement]):
            try:
                row = self.lookup_table.loc[az]
                cols_not_nan = row.index[row.notna()].tolist()
                elev = cols_not_nan[np.argmin(abs(cols_not_nan-elev))]
                measurement = self.lookup_table.at[az,elev]
            except:
                raise ValueError(f'The position of {az},{elev} is not available')
        res,h = self._conv_hrtf_mesurment(wav,fs,measurement)
        return torch.tensor(res).T,torch.tensor(h).T,int(az),int(elev)
    
    def conv_file(self,wav_path,az,elev):
        #loads wav file and conv with desierd hrtf, returns stereo audio
        wav,fs = sf.read(wav_path)
        res,hrtf,az,elev =  self.conv_hrtf_pos(wav,fs,az,elev)
        return res,fs,hrtf,az,elev

    def get_hrtf(self,az,elev,fs=16000):
        if az not in self.az_list: 
            az = self.az_list[np.argmin(abs(self.az_list-az))]
            print('not')
        if elev not in self.elev_list:
            elev = self.elev_list[np.argmin(abs(self.elev_list-elev))]
        measurement = self.lookup_table.at[az,elev]
        if np.isnan([measurement]):
            raise ValueError(f'The position of {az},{elev} is not available')
        h = np.zeros([self.hrtf_obj.Dimensions.N,2])
        h[:,0] = self.hrtf_obj.Data.IR.get_values(indices={"M":measurement, "R":0, "E":0})
        h[:,1] = self.hrtf_obj.Data.IR.get_values(indices={"M":measurement, "R":1, "E":0})
        if self.target_fs != fs:
            # h = librosa.core.resample(h.T,orig_sr=self.target_fs,target_sr=fs).T
            h = self.decimate_rir(h.T,self.target_fs,fs).T
        return h

class ImageSourceIR:
    def __init__(self,path):
        self.hrtf_obj = SOFA_HRTF_wrapper(path)
        self.hrtf_data = self.hrtf_obj.hrtf_obj.Data.IR.get_values()
        self.source_position =self.hrtf_obj.hrtf_obj.Source.Position.get_values(system="spherical", angle_unit="degree")
        self.R = 1.5
        self.room_dimensions = [6, 6, 2.8]
        self.receiver_coord = [3,3,1.5]

    def generate_A(self,absorption_data):
        '''
        input: absorption coeef dictionary with keys: walls, ceilings and floors
        output: 6X6 (coeef X wall) mat of the coeefs
        '''
        walls = list(absorption_data['walls'].keys())
        ceil =  list(absorption_data['ceilings'].keys())
        floors =  list(absorption_data['floors'].keys())

        A = np.zeros((6,6))
        for i in range(4):
            A[i,:] = absorption_data['walls'][random.choice(walls)]
        A[4,:] = absorption_data['ceilings'][random.choice(ceil)]
        A[5,:] = absorption_data['floors'][random.choice(floors)]
        return A.T

    def compute_surface_area(self, A):
        """
        Compute the effective absorbing area of the room surfaces.

        Parameters:
        - room_dimensions: List of 3 values specifying room dimensions.
        - A: Wall absorption coefficient matrix, Lx6 where L is the number of frequency bands.

        Returns:
        - S: Effective absorbing area.
        """
        Lx, Ly, Lz = self.room_dimensions
        wall_xz = Lx * Lz
        wall_yz = Ly * Lz
        wall_xy = Lx * Ly

        S = (wall_yz * (A[:, 0] + A[:, 1]) +
            wall_xz * (A[:, 2] + A[:, 3]) +
            wall_xy * (A[:, 4] + A[:, 5]))
        return S
    
    def spherical_to_cartesian(self,azimuth, elevation):
        """
        Convert spherical coordinates (azimuth, elevation) to cartesian coordinates.
        Assumes radius is 1 unless specified.

        Parameters:
        - azimuth: Azimuth angle in degrees
        - elevation: Elevation angle in degrees
        - radius: Radius (distance from origin)

        Returns:
        - Cartesian coordinates as a NumPy array [x, y, z].
        """
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)

        x = self.R * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = self.R * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = self.R * np.sin(elevation_rad)
        return np.stack((x, y, z), axis=-1)
    
    def interpolate_hrtf(self, desired_position, algorithm="bilinear"):
        """
        Interpolate the HRTF data to a desired position.

        Parameters:
        - hrtf_data: Known HRTF data, dimensions [NumSourcePositions x 2 x NumHRTFSamples].
        - source_position: Source positions corresponding to the known HRTF, [NumSourcePositions x 2].
        - desired_position: Desired position(s) for interpolation, [NumDesiredPositions x 2].
        - algorithm: Interpolation algorithm ("bilinear", "vbap", "nearest").

        Returns:
        - interpolated_hrtf: Interpolated HRTF data, dimensions [NumDesiredPositions x 2 x NumHRTFSamples].
        """
        source_position_3d = self.spherical_to_cartesian(self.source_position[:, 0], self.source_position[:, 1])
        desired_position_3d = self.spherical_to_cartesian(desired_position[:, 0], desired_position[:, 1])
       
        if algorithm == "nearest":
            tree = cKDTree(source_position_3d)
            _, idx = tree.query(desired_position_3d)
            interpolated_hrtf = self.hrtf_data[idx]
        elif algorithm == "bilinear":
            tree = cKDTree(source_position_3d)
            dist, idx = tree.query(desired_position_3d, k=4)
            weights = 1 / (dist + 1e-6)
            weights /= weights.sum(axis=1, keepdims=True)
            interpolated_hrtf = np.einsum('ij,ijkl->ikl', weights, self.hrtf_data[idx])
        elif algorithm == "vbap":
            raise NotImplementedError("VBAP interpolation is not implemented yet.")
        else:
            raise ValueError(f"Unknown interpolation algorithm: {algorithm}")

        return interpolated_hrtf
    
    def helper_image_source(self,source_coord, A,fs=48000,high_pass=True):
        """
        Estimate impulse response of a shoebox room.

        Parameters:
        - room_dimensions: List of 3 values specifying room dimensions.
        - receiver_coord: List of 3 values specifying receiver coordinates.
        - source_coord: List of 3 values specifying source coordinates.
        - A: Wall absorption coefficient matrix, Lx6 where L is the number of frequency bands.
        - F_vect: Vector of frequencies, length L.
        - fs: Sampling rate in Hz.
        - use_hrtf: Boolean, whether to use HRTF interpolation.
        - hrtf_data: Required if use_hrtf is True.
        - source_position: Required if use_hrtf is True.

        Returns:
        - h: Impulse response of the room.
        """
        # Reshape image_power to ensure proper dimensions for interpolation
        c = 343  # Speed of sound (m/s)
        F_vect = [125, 250, 500, 1000, 2000, 4000]
        x, y, z = source_coord
        use_hrtf=True
        source_xyz = np.array([
            [-x, -y, -z], [-x, -y, z], [-x, y, -z], [-x, y, z],
            [x, -y, -z], [x, -y, z], [x, y, -z], [x, y, z]
        ]).T

        Lx, Ly, Lz = self.room_dimensions
        V = Lx * Ly * Lz

        S = self.compute_surface_area(A)

        RT60 = (55.25 / c) * V / S
        imp_res_length = int(max(RT60) * fs)

        imp_res_range = c * (1 / fs) * imp_res_length
        n_max = min(int(np.ceil(imp_res_range / (2 * Lx))), 10)
        l_max = min(int(np.ceil(imp_res_range / (2 * Ly))), 10)
        m_max = min(int(np.ceil(imp_res_range / (2 * Lz))), 10)

        # highpass filter
        W = 2 * np.pi * 100 / fs  # cut-off freq (100 Hz)
        R1 = np.exp(-W)
        B1 = 2 * R1 * np.cos(W)
        B2 = -R1 * R1
        A1 = -(1 + R1)
        #### 

        B = np.sqrt(1 - A)
        bx1, bx2, by1, by2, bz1, bz2 = B.T

        surface_coeff = np.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ])
        q, j, k = surface_coeff.T

        fft_length = 512
        half_length = fft_length // 2
        one_sided_length = half_length + 1
        win = np.hanning(fft_length + 1)

        F_vect2 = np.concatenate(([0], F_vect, [fs / 2]))

        h = np.zeros((imp_res_length, 2))

        for n in range(-n_max, n_max + 1):
            Lxn2 = n * 2 * Lx
            for l in range(-l_max, l_max + 1):
                Lyl2 = l * 2 * Ly

                if use_hrtf:
                    images_vals = np.zeros((fft_length + self.hrtf_data.shape[2], 2, 2 * l_max + 1, 8))
                else:
                    images_vals = np.zeros((fft_length + 1, 2, 2 * l_max + 1, 8))

                for m_ind in range(1, 2 * m_max + 2):
                    m = m_ind - m_max - 1
                    Lzm2 = m * 2 * Lz
                    xyz = np.array([Lxn2, Lyl2, Lzm2])

                    isource_coord_v = xyz[:, None] - source_xyz
                    xyz_v = isource_coord_v - np.array(self.receiver_coord)[:, None]
                    dist_v = np.linalg.norm(xyz_v, axis=0)
                    delay_v = (fs / c) * dist_v

                    image_power = (
                            bx1[:, None] ** np.abs(n - q) *
                            by1[:, None] ** np.abs(l - j) *
                            bz1[:, None] ** np.abs(m - k) *
                            bx2[:, None] ** np.abs(n) *
                            by2[:, None] ** np.abs(l) *
                            bz2[:, None] ** np.abs(m))
                    image_power2 = np.vstack([image_power[0], image_power, image_power[-1]])
                    image_power2 = image_power2 / dist_v

                    valid_delay = delay_v <= imp_res_length
                    if not np.any(valid_delay):
                        continue
                    
                    image_power_interp = np.array([
                        interp1d(F_vect2 / (fs / 2), image_power2[:,i], kind='linear', fill_value="extrapolate")(np.linspace(0, 1, 257))
                        for i in range(8)])
                    image_power_fft = np.concatenate([image_power_interp, np.conj(image_power_interp[ :,half_length:1:-1])],axis=1) 
                    h_image_power = np.real(np.fft.ifft(image_power_fft,fft_length))
                    h_image_power = np.concatenate([h_image_power[:,one_sided_length-1:fft_length],h_image_power[:,:one_sided_length]],axis=1)
                    h_image_power *= win

                    if use_hrtf:
                        hyp = np.sqrt(xyz_v[0, :] ** 2 + xyz_v[1, :] ** 2)
                        elevation = np.arctan2(xyz_v[2, :], hyp + np.finfo(float).eps)
                        azimuth = np.arctan2(xyz_v[1, :], xyz_v[0, :])
                        azimuth *= 180 / np.pi
                        elevation *= 180 / np.pi
                        desired_position = np.stack([azimuth, elevation,np.ones_like(azimuth)*self.R], axis=-1) 
                        interpolated_ir = self.interpolate_hrtf(desired_position, algorithm="bilinear")

                        for index in range(8):
                            hrir0 = interpolated_ir[index, :, :].T 
                            hrir = np.concatenate([hrir0,np.zeros((half_length*2,2))],axis=0)
                            for ear in range(2):
                                result = lfilter(h_image_power[index,:], 1,hrir[:, ear])
                                if (m_ind-1) >= images_vals.shape[2]:
                                    images_vals = np.pad(images_vals,pad_width=((0, 0), (0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
                                images_vals[:, ear, m_ind - 1, index] = result
                    else:
                        for index in range(8):
                            for ear in range(2):
                                images_vals[:, ear, m_ind - 1, index] = h_image_power[index,:]

                    adjust_delay = np.round(delay_v).astype(int) - half_length +1

                    for index3 in range(valid_delay.shape[0]):
                        if valid_delay[index3]:
                            start = max(adjust_delay[index3] + 1, 0)
                            stop = min(adjust_delay[index3]+1+images_vals.shape[0],imp_res_length)#start + fft_length
                            h[start:stop, :] += images_vals[:stop - start, :, m_ind-1 , index3]
        if high_pass:
            for ear in range(2):
                Y = np.zeros(3)
                for idx in range(imp_res_length):
                    X0 = h[idx,ear] 
                    Y[2] = Y[1]
                    Y[1] = Y[0]
                    Y[0] = B1 * Y[1] + B2 * Y[2] + X0
                    h[idx,ear] = Y[0] + A1 * Y[1] + R1 * Y[2]
        h = h / max(np.max(np.abs(h),axis=0))
        return h[:,[1,0]]
    
    def a_to_coords_az(self,a,R,elev,receiver_coord):
        if a>=0 and a<=180:
            az = a
        else:
            az =360+a
        x = receiver_coord[0] + R*cosd(a)
        y = receiver_coord[1] - R*sind(a)
        z = receiver_coord[2] + R*sind(elev)
        return x,y,z,az




    def impzest(self,input_signal, output_signal):
        if np.any(np.isnan(input_signal)) or np.any(np.isnan(output_signal)):
            print("Warning: NaN values detected in signals")
        rir_est, _ = signal.deconvolve(output_signal, input_signal)
        return rir_est
    
    def decimate_rir(self,rir_orig, fs,target_fs=16000):
        rir_orig = np.nan_to_num(rir_orig)
        t = np.linspace(0, 3, int(3 * fs), endpoint=False)
        exc = signal.chirp(t, f0=20, f1=20000, t1=3, method='logarithmic')
        rir_dec = np.zeros_like(rir_orig)
        for i in range(rir_orig.shape[0]):
            exc = np.append(exc, np.zeros(len(rir_orig[i,:]) + 1))

            # Convolve with RIR
            sig = signal.fftconvolve(exc, rir_orig[i,:], mode='full')

            # Decimate the signal
            dec = fs // np.gcd(target_fs, fs)
            exc_dec = signal.decimate(exc, dec)
            exc_dec[np.abs(exc_dec) < 1e-10] = 0

            sig_dec = signal.decimate(sig, dec)
            res = self.impzest(exc_dec, sig_dec)
            rir_dec[i,:len(res)] = res
        return rir_dec
    
    def get_rt60(self,A_mat):
        c=343
        V = np.prod(self.room_dimensions)
        S = self.compute_surface_area(A_mat)
        RT60 = (55.25 / c) * V / S
        return RT60
    
    def get_hrtf_ir(self,a,elev,A_mat=None):
        c=343
        if type(A_mat) == type(None):
            A = self.generate_A(absorption_data)
        else:
            A =A_mat
        V = np.prod(self.room_dimensions)
        S = self.compute_surface_area(A)
        RT60 = (55.25 / c) * V / S
        x,y,z,_ =self.a_to_coords_az(a,self.R,elev,self.receiver_coord)
        source_coord = [x,y,z]
        h = self.helper_image_source( source_coord, A)
        return h,RT60,A
    
    def _conv_file_ir(self,wav_path,h,fs_h=48000):
        wav,sr = sf.read(wav_path)
        if sr == 16000:
            # h = h[::3,:]
            h = self.decimate_rir(h,fs_h,sr)
        # if sr < fs_h:
        #     h = self.decimate_rir(h,fs_h,sr)
        # elif sr > fs_h:
        #     print('wrong sampling rates of IR')
        #     exit()
        rend_L = signal.fftconvolve(wav,h[:,0])
        rend_R = signal.fftconvolve(wav,h[:,1])
        stereo_audio = np.concatenate((rend_L[:,np.newaxis],rend_R[:,np.newaxis]),axis=1)
        stereo_audio = stereo_audio/np.max(abs(stereo_audio))
        return stereo_audio

    
if __name__=="__main__":
    hrtf_ir = ImageSourceIR('/home/workspace/yoavellinson/extraction_master/hrtf_testing/RIEC_hrir_subject_001.sofa')
    azs = {}
    tmp_az = 90
    for i in range(37): #create a dictionary of all mesurments
        print(tmp_az)
        d = {'az':tmp_az}
        azs[str(i)] = d
        tmp_az -=5
        if tmp_az ==-5:
            tmp_az = 355

    for i,d in azs.items():
        hrtf_1 = hrtf_ir.hrtf_obj.get_hrtf(d['az'],0)
        sf.write(f'/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/blcmv_elior_roi/hrtf_az_{d['az']}.wav',hrtf_1,16000)

#     hrtf_2 = hrtf_ir.hrtf_obj.get_hrtf(a2,0)
#     sf.write(f'hrtf_{a2}.wav',hrtf_2,16000)
#     A = np.array([[0.53, 0.92, 1.00, 1.00, 1.00, 1.00],
#         [0.53, 0.92, 1.00, 1.00, 1.00, 1.00],
#         [0.53, 0.92, 1.00, 1.00, 1.00, 1.00],
#         [0.53, 0.92, 1.00, 1.00, 1.00, 1.00],
#         [0.45, 0.80, 0.90, 0.90, 0.90, 0.80],
#         [0.15, 0.25, 0.50, 0.60, 0.70, 0.70]]).T

#     h1,rt60,_ = hrtf_ir.get_hrtf_ir(a1,0,A)
#     h2,_,_ = hrtf_ir.get_hrtf_ir(a2,0,A)
#     sf.write('h1.wav',h1,48000)
#     sf.write('h2.wav',h2,48000)
#     print(rt60)
#     print('Done with IR')
#     wav_path1 = '/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/test-clean/5683/32866/5683-32866-0014.wav'
#     wav_path2 = '/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/test-clean/260/123440/260-123440-0004.wav'
#     wav_1_ir = hrtf_ir._conv_file_ir(wav_path1,h1)
#     wav_2_ir = hrtf_ir._conv_file_ir(wav_path2,h2)
#     mix = np.zeros((int(max(wav_1_ir.shape[0],wav_2_ir.shape[0])),2))
#     mix[:wav_1_ir.shape[0],:] += wav_1_ir
#     mix[:wav_2_ir.shape[0],:] += wav_2_ir
#     mix = mix/(np.abs(mix).max(0))
#     # wav = wav/(wav.abs().max(dim=-1, keepdim=True).values)
#     sf.write('mix_test.wav',mix,16000)
# #     h = SOFA_HRTF_wrapper('Kayser2009_Anechoic.sofa')#('/home/workspace/yoavellinson/extraction_master/hrtf_testing/RIEC_hrir_subject_001.sofa')
# #     print(h.lookup_table)
# #     # for a in range(0,360,10):
# #     #     s,fs,hrtf,az,elev = h.conv_file('/dsi/gannot-lab2/datasets2/LRS2/audio_only/main/5983607991811548308/00034.wav',a,0)
# #     #     sf.write(f'/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/res_hrtf_{a}.wav',s.T,fs)
        
