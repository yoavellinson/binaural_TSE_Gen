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
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/test_set/viking/subj_O.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.2 0.5 0.87 0.99 1.0 1.0;
                                    0.18 0.42 0.55 0.77 0.87 0.94;
                                    0.1 0.1 0.15 0.2 0.3 0.3;
                                    0.2 0.55 1.0 1.0 1.0 1.0;
                                    0.4 0.35 0.2 0.15 0.05 0.05;
                                    0.1 0.25 0.3 0.3 0.3 0.3];
receiver(1).orientation      = [ 48.403320042936265 0 0 ];
receiver(1).location         = [ 3.5141946696254602 3.212361840694974 1.7065768506878527 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 2.443452702801781 2.63333363248344 1.8130747429445995 ];
source(1).orientation        = [ -111.59667995706374 -5.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 4.707274417314142 2.9675297861251706 1.3802303519286394 ];
source(2).orientation        = [ 168.40332004293626 15.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 10 10 10 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf_1k_mp/viking_subj_O_az1160.0_elev15.0_az2300.0_elev2-15.0/rt_60_0.3614204658175551/h_rt60';
