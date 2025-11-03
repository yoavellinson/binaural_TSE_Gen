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
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/test_set/sadie/H7_48K_24bit_256tap_FIR_SOFA.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.11 0.32 0.56 0.77 0.9 0.94;
                                    0.12 0.56 0.96 1.0 1.0 1.0;
                                    0.12 0.1 0.08 0.07 0.07 0.07;
                                    0.18 0.42 0.55 0.77 0.87 0.94;
                                    0.42 0.72 0.83 0.88 0.89 0.8;
                                    0.1 0.25 0.3 0.3 0.3 0.3];
receiver(1).orientation      = [ 48.403320042936265 0 0 ];
receiver(1).location         = [ 3.5141946696254602 3.212361840694974 1.7065768506878527 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 4.424518525066569 2.4042328264670867 1.1389515066114528 ];
source(1).orientation        = [ 138.40332004293626 25.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 4.680746358664094 3.562417802065256 2.032923349447066 ];
source(2).orientation        = [ 196.70332004293627 -15.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 0 0 0 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf_1k_mp/sadie_H7_48K_24bit_256tap_FIR_SOFA_az1270.0_elev1-25.0_az2328.3_elev215.0/rt_60_0.239451988884461/h_first';
