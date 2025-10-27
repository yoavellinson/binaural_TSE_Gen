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
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/sadie/D2_48K_24bit_256tap_FIR_SOFA.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.12 0.1 0.08 0.07 0.07 0.07;
                                    0.3 0.0 0.3 0.3 0.3 0.3;
                                    0.2 0.62 0.98 0.62 0.62 0.62;
                                    0.06 0.1 0.08 0.09 0.07 0.05;
                                    0.42 0.72 0.83 0.88 0.89 0.8;
                                    0.1 0.25 0.3 0.3 0.3 0.3];
receiver(1).orientation      = [ 63.547676231127674 0 0 ];
receiver(1).location         = [ 3.5867942650630993 5.4747659439790635 1.5918663507651607 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 4.210660599868375 6.912831457332441 2.0118925535959575 ];
source(1).orientation        = [ -113.45232376887233 -15.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 2.647533177468499 6.367723039720181 1.183243041499285 ];
source(2).orientation        = [ -43.55232376887232 17.5 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 10 10 10 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf/sadie_D2_48K_24bit_256tap_FIR_SOFA_az13.0_elev115.0_az272.9_elev2-17.5/rt_60_0.30649111193413847/h_rt60';
