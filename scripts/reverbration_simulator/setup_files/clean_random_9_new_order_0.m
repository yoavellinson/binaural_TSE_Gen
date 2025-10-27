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
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/sadie/H19_48K_24bit_256tap_FIR_SOFA.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.03 0.04 0.03 0.03 0.03 0.03;
                                    0.05 0.1 0.2 0.55 0.6 0.55;
                                    0.15 0.2 0.1 0.1 0.1 0.1;
                                    0.12 0.1 0.08 0.07 0.07 0.07;
                                    0.3 0.2 0.15 0.05 0.05 0.05;
                                    0.14 0.1 0.1 0.1 0.1 0.1];
receiver(1).orientation      = [ 43.049260454466776 0 0 ];
receiver(1).location         = [ 5.220642062547402 6.777766357359335 1.7678905703238075 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 5.83142204379336 7.881880262519695 2.1657321409269787 ];
source(1).orientation        = [ -118.95073954553322 -17.5 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 6.45287892000203 7.777367454352641 1.3427361662641215 ];
source(2).orientation        = [ 219.04926045446678 15.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 0 0 0 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf/sadie_H19_48K_24bit_256tap_FIR_SOFA_az118.0_elev117.5_az2356.0_elev2-15.0/rt_60_1.2369889143855306/h_first';
