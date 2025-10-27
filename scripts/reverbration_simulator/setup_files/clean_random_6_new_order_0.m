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
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/sadie/H8_48K_24bit_256tap_FIR_SOFA.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.03 0.04 0.03 0.03 0.03 0.03;
                                    0.43 0.69 0.98 1.0 1.0 1.0;
                                    0.37 0.85 1.0 1.0 1.0 1.0;
                                    0.2 0.55 1.0 1.0 1.0 1.0;
                                    0.3 0.7 0.85 0.9 0.85 0.8;
                                    0.02 0.02 0.04 0.05 0.05 0.1];
receiver(1).orientation      = [ 11.813894070355921 0 0 ];
receiver(1).location         = [ 3.4560631387978926 3.729799589155225 1.6330587830205956 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 4.373690289897534 4.5799450744155115 2.027470590735023 ];
source(1).orientation        = [ -137.18610592964407 -17.5 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 4.844690108489885 3.9950433588279193 1.6330587830205956 ];
source(2).orientation        = [ 190.81389407035593 -0.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 0 0 0 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf/sadie_H8_48K_24bit_256tap_FIR_SOFA_az131.0_elev117.5_az2359.0_elev20.0/rt_60_0.25674939584837847/h_first';
