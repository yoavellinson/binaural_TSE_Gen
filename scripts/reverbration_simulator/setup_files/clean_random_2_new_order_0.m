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
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/axd/p0103.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.1 0.3 0.6 0.7 0.7 0.6;
                                    0.2 0.62 0.98 0.62 0.62 0.62;
                                    0.05 0.45 0.7 0.85 0.9 0.85;
                                    0.1 0.2 0.45 0.8 0.8 0.75;
                                    0.4 0.35 0.2 0.15 0.05 0.05;
                                    0.01 0.01 0.01 0.01 0.02 0.02];
receiver(1).orientation      = [ 42.07162120867529 0 0 ];
receiver(1).location         = [ 4.0981853310789536 4.642964094443725 1.6518223898089914 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 4.047018494477366 6.057492074528496 1.4022398200139223 ];
source(1).orientation        = [ -87.92837879132472 10.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 5.470391943226562 4.327971434916322 1.6518223898089914 ];
source(2).orientation        = [ 167.07162120867528 -0.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 0 0 0 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf/axd_p0103_az150.0_elev1-10.0_az2305.0_elev20.0/rt_60_0.48374819772197375/h_first';
