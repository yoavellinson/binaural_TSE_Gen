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
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/fhk/HRIR_CIRC360_NF100.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.15 0.2 0.1 0.1 0.1 0.1;
                                    0.34 0.68 0.94 1.0 1.0 1.0;
                                    0.2 0.55 1.0 1.0 1.0 1.0;
                                    0.05 0.45 0.7 0.85 0.9 0.85;
                                    0.3 0.2 0.15 0.05 0.05 0.05;
                                    0.04 0.1 0.07 0.06 0.07 0.07];
receiver(1).orientation      = [ 77.30613925297969 0 0 ];
receiver(1).location         = [ 4.252439152029396 5.245557569720539 1.7349092858093549 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 4.46237686046638 6.799544377952789 1.7349092858093549 ];
source(1).orientation        = [ -97.69386074702031 -0.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 3.7027063037386063 6.462733114536153 1.7349092858093549 ];
source(2).orientation        = [ -65.69386074702031 -0.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 10 10 10 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf/fhk_HRIR_CIRC360_NF100_az15.0_elev10.0_az237.0_elev20.0/rt_60_0.47132660436310947/h_rt60';
