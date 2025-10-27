room.humidity               = 0.42;         % relative humidity (0,...,1)
room.temperature            = 20;           % room temperature (celsius)
room.surface.frequency      = [  125       250       500       1000      2000      4000];


room.surface.diffusion      = [  0.0       0.0       0.0       0.0       0.0       0.0   ; 
                                 0.0       0.0       0.0       0.0       0.0       0.0   ; 
                                 0.0       0.0       0.0       0.0       0.0       0.0   ; 
                                 0.0       0.0       0.0       0.0       0.0       0.0   ; 
                                 0.0       0.0       0.0       0.0       0.0       0.0   ; 
                                 0.0       0.0       0.0       0.0       0.0       0.0    ];

options.fs                  = 16000;                % sampling frequency in Hz
options.responseduration    = 1.8;                 % duration of impulse response
options.bandsperoctave      = 4;                    % simulation frequency accuracy (1, 2, 3, or 4 bands/octave)
options.referencefrequency  = 125;                  % reference frequency for frequency octaves
options.airabsorption       = true;                 % apply air absorption?
options.distanceattenuation = true;                 % apply distance attenuation?
options.subsampleaccuracy   = true;                % apply subsample accuracy?
options.highpasscutoff      = 1;                    % 3dB frequency of high-pass filter (0=none)
options.verbose             = false;                 % print status messages?

options.simulatespecular    = true;                 % simulate specular reflections?

options.simulatediffuse     = false;                 % simulate diffuse reflections?
options.numberofrays        = 2000;                 % number of rays in simulation (20*K^2)
options.diffusetimestep     = 0.010;                % time resolution in diffuse energy histogram (seconds)
options.rayenergyfloordB    = -80;                  % ray energy threshold (dB, with respect to initial energy)
options.uncorrelatednoise   = true;                 % use uncorrelated poisson arrivals for binaural impulse responses?

options.mex_saveaswav       = false;                % enable or disable saving the results of sofamyroom on disk
                                                    % when using MATLAB
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/ari_atl_and_full/hrtf_b_nh780.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.09 0.22 0.54 0.76 0.88 0.93;
                                    0.1 0.1 0.15 0.2 0.3 0.3;
                                    0.12 0.04 0.06 0.05 0.05 0.05;
                                    0.12 0.1 0.08 0.07 0.07 0.07;
                                    0.45 0.8 0.9 0.9 0.9 0.8;
                                    0.04 0.1 0.07 0.06 0.07 0.07];
receiver(1).orientation      = [ 60.49025346698263 0 0 ];
receiver(1).location         = [ 5.137434465105406 5.341280493928155 1.5743060643738493 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 6.160422685404195 6.214694059201659 1.8114875593252673 ];
source(1).orientation        = [ 220.49025346698264 -10.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 3.8467423566439365 6.0718088206886325 1.17691313334132 ];
source(2).orientation        = [ -29.50974653301737 15.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 0 0 0 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf/ari_atl_and_full_hrtf_b_nh780_az1340.0_elev110.0_az290.0_elev2-15.0/rt_60_0.3579657855285099/h_first';
