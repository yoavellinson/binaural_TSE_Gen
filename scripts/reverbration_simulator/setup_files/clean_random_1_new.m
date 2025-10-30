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
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/test_set/ari_atl_and_full/hrtf_las_nh919.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.1 0.1 0.15 0.2 0.3 0.3;
                                    0.03 0.04 0.03 0.03 0.03 0.03;
                                    0.03 0.05 0.17 0.52 0.52 0.52;
                                    0.3 0.3 0.6 0.85 0.75 0.75;
                                    0.45 0.8 0.9 0.9 0.9 0.8;
                                    0.01 0.01 0.01 0.01 0.02 0.02];
receiver(1).orientation      = [ 42.6291968823217 0 0 ];
receiver(1).location         = [ 6.967499802747377 6.342418151993132 1.709678678567706 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 6.155595130796888 5.061618924434697 2.288756544589343 ];
source(1).orientation        = [ -152.3708031176783 -20.9 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 7.858963107437428 7.468343088456293 1.709678678567706 ];
source(2).orientation        = [ -128.3708031176783 -0.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 10 10 10 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf_1k_mp/ari_atl_and_full_hrtf_las_nh919_az1195.0_elev120.9_az29.0_elev20.0/rt_60_0.3502573576651493/h_rt60';
