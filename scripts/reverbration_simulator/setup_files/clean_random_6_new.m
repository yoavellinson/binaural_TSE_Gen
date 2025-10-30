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
                                    0.2 0.62 0.98 0.62 0.62 0.62;
                                    0.05 0.45 0.7 0.85 0.9 0.85;
                                    0.1 0.1 0.15 0.2 0.3 0.3;
                                    0.45 0.8 0.9 0.9 0.9 0.8;
                                    0.15 0.25 0.5 0.6 0.7 0.7];
receiver(1).orientation      = [ 42.6291968823217 0 0 ];
receiver(1).location         = [ 6.967499802747377 6.342418151993132 1.709678678567706 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 8.424996746419579 6.761152574790575 1.709678678567706 ];
source(1).orientation        = [ 196.02919688232168 -0.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 8.3214813923476 5.863724469997918 1.709678678567706 ];
source(2).orientation        = [ 160.52919688232168 -0.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 10 10 10 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf_1k_mp/sadie_H7_48K_24bit_256tap_FIR_SOFA_az1333.4_elev10.0_az2297.9_elev20.0/rt_60_0.2038021949825969/h_rt60';
