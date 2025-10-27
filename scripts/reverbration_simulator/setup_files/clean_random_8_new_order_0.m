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
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/fhk/HRIR_L2354.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.34 0.68 0.94 1.0 1.0 1.0;
                                    0.05 0.0 0.15 0.0 0.3 0.3;
                                    0.2 0.55 1.0 1.0 1.0 1.0;
                                    0.37 0.85 1.0 1.0 1.0 1.0;
                                    0.3 0.7 0.85 0.9 0.85 0.8;
                                    0.03 0.09 0.25 0.31 0.33 0.44];
receiver(1).orientation      = [ 36.44763273144143 0 0 ];
receiver(1).location         = [ 4.325519950358768 3.1729382414604497 1.6921214143279524 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 5.124467546224882 4.42450895329541 1.6921214143279524 ];
source(1).orientation        = [ -122.55236726855857 -0.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 5.798982837347786 3.0036940090306 1.6921214143279524 ];
source(2).orientation        = [ 173.44763273144144 -0.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 0 0 0 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf/fhk_HRIR_L2354_az121.0_elev10.0_az2317.0_elev20.0/rt_60_0.21807415378986034/h_first';
