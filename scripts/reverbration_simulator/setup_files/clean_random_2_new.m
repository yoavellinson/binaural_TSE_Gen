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
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/test_set/ss2/JLG200080922_1_processed.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.34 0.67 0.98 1.0 1.0 1.0;
                                    0.2 0.5 0.87 0.99 1.0 1.0;
                                    0.03 0.05 0.17 0.52 0.52 0.52;
                                    0.2 0.62 0.98 0.62 0.62 0.62;
                                    0.3 0.2 0.15 0.05 0.05 0.05;
                                    0.03 0.09 0.25 0.31 0.33 0.44];
receiver(1).orientation      = [ 48.403320042936265 0 0 ];
receiver(1).location         = [ 3.5141946696254602 3.212361840694974 1.7065768506878527 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 2.5243851795339705 3.9209090859418425 2.3268101956200136 ];
source(1).orientation        = [ -47.596679957063735 -27.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 3.0109932820399337 4.321492303184289 1.8994798674459143 ];
source(2).orientation        = [ -65.59667995706374 -9.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 10 10 10 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf_1k_mp/ss2_JLG200080922_1_processed_az196.0_elev127.0_az266.0_elev29.0/rt_60_0.3540239438756706/h_rt60';
