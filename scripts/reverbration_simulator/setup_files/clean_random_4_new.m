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
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/ari_atl_and_full/hrtf_b_nh231.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.05 0.45 0.7 0.85 0.9 0.85;
                                    0.53 0.92 1.0 1.0 1.0 1.0;
                                    0.34 0.68 0.94 1.0 1.0 1.0;
                                    0.37 0.85 1.0 1.0 1.0 1.0;
                                    0.42 0.72 0.83 0.88 0.89 0.8;
                                    0.15 0.25 0.5 0.6 0.7 0.7];
receiver(1).orientation      = [ 22.974750654699886 0 0 ];
receiver(1).location         = [ 5.938333846758079 5.667288345541202 1.544304630846409 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 6.711959319604318 4.970095274529747 1.544304630846409 ];
source(1).orientation        = [ 137.97475065469987 -0.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 6.723693764947064 6.538746286426927 1.2299659489637325 ];
source(2).orientation        = [ -132.02524934530013 15.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 10 10 10 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf/ari_atl_and_full_hrtf_b_nh231_az1295.0_elev10.0_az225.0_elev2-15.0/rt_60_0.17785549071503287/h_rt60';
