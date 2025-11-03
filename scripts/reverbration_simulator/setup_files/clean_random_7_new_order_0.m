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
room.surface.absorption  = [0.03 0.05 0.17 0.52 0.52 0.52;
                                    0.37 0.85 1.0 1.0 1.0 1.0;
                                    0.1 0.1 0.15 0.2 0.3 0.3;
                                    0.11 0.32 0.56 0.77 0.9 0.94;
                                    0.45 0.8 0.9 0.9 0.9 0.8;
                                    0.15 0.25 0.5 0.6 0.7 0.7];
receiver(1).orientation      = [ 48.403320042936265 0 0 ];
receiver(1).location         = [ 3.5141946696254602 3.212361840694974 1.7065768506878527 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 4.731440989611644 3.2209304988276144 2.173845511527875 ];
source(1).orientation        = [ 180.40332004293626 -21.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 3.145993038086133 2.051409827120968 1.3802303519286394 ];
source(2).orientation        = [ -155.59667995706374 15.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 0 0 0 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf_1k_mp/ss2_JLG200080922_1_processed_az1312.0_elev121.0_az2204.0_elev2-15.0/rt_60_0.20720501640024597/h_first';
